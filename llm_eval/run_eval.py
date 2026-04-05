import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import DefaultDict, Dict, List

# Allow running from any working directory (e.g., from a cloud notebook).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tqdm import tqdm

from client import HuggingFaceLocalClient, OpenAICompatibleClient
from data_loader import load_dataset
from metrics import compute_coqe_metrics, leaderboard_row, metrics_to_lines, CAMERA_COQE_LABEL_ORDER, VCOM_LABEL_ORDER
from prompts import build_messages


_PRED_TUPLE_RE = re.compile(
    r"\[S\]\s*(.*?)\s*\[O\]\s*(.*?)\s*\[A\]\s*(.*?)\s*\[P\]\s*(.*?)\s*\[L\]\s*(.*?)(?=\)|\n|;|$)",
    re.DOTALL,
)

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _parse_pred_tuples(text: str) -> List[Dict[str, str]]:
    tuples: List[Dict[str, str]] = []
    for part in (text or "").split(";"):
        m = _PRED_TUPLE_RE.search(part.strip().strip("()"))
        if not m:
            continue
        s, o, a, p, l = (x.strip() for x in m.groups())
        tuples.append({"S": s, "O": o, "A": a, "P": p, "L": l})
    return tuples


def _is_unk_or_empty(text: str) -> bool:
    t = (text or "").strip()
    return t == "" or t == "[UNK]"


def _tokenize_with_offsets(text: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for m in _WORD_RE.finditer(text):
        tok = m.group(0)
        out.append({
            "raw": tok,
            "norm": tok.lower(),
            "start": m.start(),
            "end": m.end(),
        })
    return out


def _span_token_positions_from_tokenized(slot_value: str, tokenized_tokens: List[str]) -> List[int]:
    """Find positions of slot_value in pre-tokenized tokens (1-indexed).
    
    tokenized_tokens: list of tokens from tokenized sentence (split by whitespace).
    Returns: 1-indexed positions of matching tokens.
    """
    slot_value = (slot_value or "").strip()
    if not slot_value or not tokenized_tokens:
        return []
    
    slot_tokens = slot_value.split()
    m = len(slot_tokens)
    
    # 1) Exact contiguous match
    for i in range(len(tokenized_tokens) - m + 1):
        if tokenized_tokens[i:i+m] == slot_tokens:
            return list(range(i + 1, i + m + 1))
    
    # 2) Non-contiguous greedy fallback
    positions = []
    cursor = 0
    for token in slot_tokens:
        found = -1
        for j in range(cursor, len(tokenized_tokens)):
            if tokenized_tokens[j] == token:
                found = j
                break
        if found < 0:
            # Try full scan if not found from cursor
            for j in range(0, len(tokenized_tokens)):
                if tokenized_tokens[j] == token:
                    found = j
                    break
        if found < 0:
            return []
        positions.append(found + 1)
        cursor = found + 1
    return positions


def _span_token_positions(sentence: str, span_text: str) -> List[int]:
    span_text = (span_text or "").strip()
    if _is_unk_or_empty(span_text):
        return []

    sent_tokens = _tokenize_with_offsets(sentence)
    span_tokens = _tokenize_with_offsets(span_text)
    if not sent_tokens or not span_tokens:
        return []

    sent_norm = [t["norm"] for t in sent_tokens]
    span_norm = [t["norm"] for t in span_tokens]
    m = len(span_norm)

    # 1) exact contiguous token match
    for i in range(0, len(sent_norm) - m + 1):
        if sent_norm[i : i + m] == span_norm:
            return list(range(i + 1, i + m + 1))

    # 2) character-level fallback then project to token range
    idx = sentence.lower().find(span_text.lower())
    if idx >= 0:
        end = idx + len(span_text)
        positions = []
        for i, tok in enumerate(sent_tokens, start=1):
            if not (tok["end"] <= idx or tok["start"] >= end):
                positions.append(i)
        if positions:
            return positions

    # 3) non-contiguous greedy fallback
    positions = []
    cursor = 0
    for w in span_norm:
        found = -1
        for j in range(cursor, len(sent_norm)):
            if sent_norm[j] == w:
                found = j
                break
        if found < 0:
            for j in range(0, len(sent_norm)):
                if sent_norm[j] == w:
                    found = j
                    break
        if found < 0:
            return []
        positions.append(found + 1)
        cursor = found + 1
    return positions


def _to_indexed_slot(slot_value: str, sentence: str, tokenized_tokens: List[str] = None) -> str:
    if _is_unk_or_empty(slot_value):
        return ""
    
    # If tokenized_tokens provided, use them for accurate index calculation
    if tokenized_tokens:
        positions = _span_token_positions_from_tokenized(slot_value, tokenized_tokens)
        if not positions:
            return ""
        slot_tokens = slot_value.split()
        if len(slot_tokens) != len(positions):
            n = min(len(slot_tokens), len(positions))
            slot_tokens = slot_tokens[:n]
            positions = positions[:n]
        return " ".join(f"{pos}&&{tok}" for pos, tok in zip(positions, slot_tokens))
    
    # Fallback to original behavior (character-level) if no tokenized_tokens
    positions = _span_token_positions(sentence, slot_value)
    if not positions:
        return ""
    slot_tokens = _tokenize_with_offsets(slot_value)
    if len(slot_tokens) != len(positions):
        # Keep alignment stable by trimming to the shorter side.
        n = min(len(slot_tokens), len(positions))
        slot_tokens = slot_tokens[:n]
        positions = positions[:n]
    return " ".join(f"{pos}&&{tok['raw']}" for pos, tok in zip(positions, slot_tokens))


def _map_label_for_camera(label: str) -> str:
    # camera-coqe raw labels: -1(worse), 0(equal), 1(better), 2(different)
    m = {
        "Worse": "-1",
        "Equal": "0",
        "Better": "1",
        "Different": "2",
        "[UNK]": "",
        "": "",
    }
    return m.get((label or "").strip(), "")


def _save_predictions_raw_camera(file_path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as fp:
        for row in rows:
            sent = row["input"]
            pred = row["prediction"]
            tuples = _parse_pred_tuples(pred)

            valid = []
            for t in tuples:
                if all(_is_unk_or_empty(t[k]) for k in ("S", "O", "A", "P", "L")):
                    continue
                valid.append(t)

            flag = "1" if valid else "0"
            fp.write(f"{sent}\t{flag}\n")

            if not valid:
                fp.write("[[];[];[];[];[]]\n")
                continue

            for t in valid:
                s = _to_indexed_slot(t["S"], sent)
                o = _to_indexed_slot(t["O"], sent)
                a = _to_indexed_slot(t["A"], sent)
                p = _to_indexed_slot(t["P"], sent)
                l = _map_label_for_camera(t["L"])
                fp.write(f"[[{s}];[{o}];[{a}];[{p}];[{l}]]\n")


def _save_predictions_raw_vcom(file_path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as fp:
        for row in rows:
            sent = row["input"]
            tokenized_sent = row.get("tokenized_input", sent)
            pred = row["prediction"]
            tuples = _parse_pred_tuples(pred)

            # Write sentence line with original and tokenized versions
            fp.write(f"{sent}\t{tokenized_sent}\n")
            
            # Split tokenized sentence for index calculation
            tokenized_tokens = tokenized_sent.split()

            written_any = False
            for t in tuples:
                if all(_is_unk_or_empty(t[k]) for k in ("S", "O", "A", "P", "L")):
                    continue

                # If model does not predict a valid label, skip this tuple.
                # This keeps sentence output consistent with raw data: sentence line then empty line.
                if _is_unk_or_empty(t["L"]):
                    continue

                # Pass tokenized_tokens for accurate index calculation
                subj_slot = _to_indexed_slot(t["S"], sent, tokenized_tokens)
                obj_slot = _to_indexed_slot(t["O"], sent, tokenized_tokens)
                asp_slot = _to_indexed_slot(t["A"], sent, tokenized_tokens)
                pre_slot = _to_indexed_slot(t["P"], sent, tokenized_tokens)

                obj = {
                    "subject": subj_slot.split() if subj_slot else [],
                    "object": obj_slot.split() if obj_slot else [],
                    "aspect": asp_slot.split() if asp_slot else [],
                    "predicate": pre_slot.split() if pre_slot else [],
                    "label": t["L"],
                }
                fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written_any = True

            fp.write("\n")


def _norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _save_predictions_raw_vcom_mapped(
    file_path: str,
    rows: List[Dict],
    datasets_root: str,
    split: str,
) -> None:
    """Write VCOM raw-format prediction file mapped to full raw split order.

    - Keeps metadata filtering in dataloader unchanged.
    - Reconstructs output over all raw sentence lines.
    - For missing/filtered sentences, writes sentence line then blank line.
    """
    split_dir = os.path.join(datasets_root, "vcom-data", split)
    if not os.path.isdir(split_dir):
        # Fallback to existing behavior if raw directory is unavailable.
        _save_predictions_raw_vcom(file_path, rows)
        return

    pred_by_sentence: DefaultDict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        pred_by_sentence[_norm_key(row.get("input", ""))].append(row)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as fp:
        for fn in sorted(x for x in os.listdir(split_dir) if x.endswith(".txt")):
            path = os.path.join(split_dir, fn)
            with open(path, "r", encoding="utf-8") as in_fp:
                for raw_line in in_fp:
                    line = raw_line.strip()
                    if not line:
                        continue
                    if "\t" not in line or line.startswith("{"):
                        continue

                    parts = line.split("\t", 1)
                    sent = parts[0].strip()
                    tokenized_sent = parts[1].strip() if len(parts) > 1 else sent

                    # Always keep raw sentence line from dataset.
                    fp.write(f"{sent}\t{tokenized_sent}\n")

                    key = _norm_key(sent)
                    if not pred_by_sentence[key]:
                        # Missing due to metadata filtering or other mismatch.
                        fp.write("\n")
                        continue

                    row = pred_by_sentence[key].pop(0)
                    pred = row.get("prediction", "")
                    tuples = _parse_pred_tuples(pred)
                    tokenized_tokens = tokenized_sent.split()

                    wrote_any = False
                    for t in tuples:
                        if all(_is_unk_or_empty(t[k]) for k in ("S", "O", "A", "P", "L")):
                            continue
                        if _is_unk_or_empty(t["L"]):
                            continue

                        subj_slot = _to_indexed_slot(t["S"], sent, tokenized_tokens)
                        obj_slot = _to_indexed_slot(t["O"], sent, tokenized_tokens)
                        asp_slot = _to_indexed_slot(t["A"], sent, tokenized_tokens)
                        pre_slot = _to_indexed_slot(t["P"], sent, tokenized_tokens)

                        obj = {
                            "subject": subj_slot.split() if subj_slot else [],
                            "object": obj_slot.split() if obj_slot else [],
                            "aspect": asp_slot.split() if asp_slot else [],
                            "predicate": pre_slot.split() if pre_slot else [],
                            "label": t["L"],
                        }
                        fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        wrote_any = True

                    if not wrote_any:
                        fp.write("\n")
                    else:
                        fp.write("\n")


def _prediction_to_indexed_vcom(pred: str, sentence: str, tokenized_sentence: str) -> str:
    """Convert model prediction tuples to index-span tuples for evaluate_v1-style scoring."""
    tuples = _parse_pred_tuples(pred)
    tokenized_tokens = tokenized_sentence.split()
    out_parts: List[str] = []

    for t in tuples:
        if all(_is_unk_or_empty(t[k]) for k in ("S", "O", "A", "P", "L")):
            continue
        s = _to_indexed_slot(t["S"], sentence, tokenized_tokens) or "[UNK]"
        o = _to_indexed_slot(t["O"], sentence, tokenized_tokens) or "[UNK]"
        a = _to_indexed_slot(t["A"], sentence, tokenized_tokens) or "[UNK]"
        p = _to_indexed_slot(t["P"], sentence, tokenized_tokens) or "[UNK]"
        l = t["L"] if not _is_unk_or_empty(t["L"]) else "[UNK]"
        out_parts.append(f"([S] {s} [O] {o} [A] {a} [P] {p} [L] {l})")

    if out_parts:
        return " ; ".join(out_parts)
    return "([S] [UNK] [O] [UNK] [A] [UNK] [P] [UNK] [L] [UNK])"


def _prediction_to_indexed_camera(pred: str, sentence: str) -> str:
    """Convert model prediction tuples to index-span tuples for camera-coqe scoring."""
    tuples = _parse_pred_tuples(pred)
    out_parts: List[str] = []

    for t in tuples:
        if all(_is_unk_or_empty(t[k]) for k in ("S", "O", "A", "P", "L")):
            continue
        s = _to_indexed_slot(t["S"], sentence) or "[UNK]"
        o = _to_indexed_slot(t["O"], sentence) or "[UNK]"
        a = _to_indexed_slot(t["A"], sentence) or "[UNK]"
        p = _to_indexed_slot(t["P"], sentence) or "[UNK]"
        l = t["L"] if not _is_unk_or_empty(t["L"]) else "[UNK]"
        out_parts.append(f"([S] {s} [O] {o} [A] {a} [P] {p} [L] {l})")

    if out_parts:
        return " ; ".join(out_parts)
    return "([S] [UNK] [O] [UNK] [A] [UNK] [P] [UNK] [L] [UNK])"

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate popular LLMs on COQE datasets")
    parser.add_argument(
        "--datasets",
        type=str,
        default="t5-camera-coqe-data,vcom-data",
        help="Comma-separated datasets: t5-camera-coqe-data,vcom-data",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model IDs (space separated), for example: openai/gpt-4o-mini anthropic/claude-3.5-sonnet",
    )
    parser.add_argument("--base-url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", type=str, default="OPENROUTER_API_KEY")
    parser.add_argument(
        "--provider",
        type=str,
        default="openrouter",
        choices=["openrouter", "hf-local"],
        help="Inference backend provider",
    )
    parser.add_argument("--hf-dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16"])
    parser.add_argument("--hf-load-in-4bit", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="Set >0 for quick smoke tests")
    parser.add_argument("--debug-samples", type=int, default=0, help="Print prompt+output for first N samples per model")
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=8,
        help="Batch size for local hf inference (provider=hf-local)",
    )
    parser.add_argument("--datasets-root", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--cache-dir", type=str, default="")
    parser.add_argument(
        "--prompt-strategy",
        type=str,
        default="zero-shot",
        choices=["zero-shot", "few-shot", "cot"],
        help="Prompting strategy: zero-shot, few-shot, or cot",
    )
    parser.add_argument(
        "--match-mode",
        type=str,
        default="index-match",
        choices=["index-match", "non-index-match"],
        help="Metric matching mode: index-match (position-aware) or non-index-match (phrase-aware)",
    )
    return parser.parse_args()


def _default_path(path_value: str, relative_path: str) -> str:
    if path_value:
        return path_value
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    return os.path.join(project_root, relative_path)


def _slugify(text: str) -> str:
    text = text.lower().replace("/", "__")
    text = re.sub(r"[^a-z0-9_\-.]+", "_", text)
    return text.strip("_")


def _load_cache(cache_file: str) -> Dict[str, str]:
    if not os.path.exists(cache_file):
        return {}

    cache = {}
    with open(cache_file, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                key = row.get("input", "")
                val = row.get("prediction", "")
                if key:
                    cache[key] = val
            except json.JSONDecodeError:
                continue
    return cache


def _append_cache(cache_file: str, sentence: str, prediction: str) -> None:
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    row = {"input": sentence, "prediction": prediction}
    with open(cache_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_predictions(file_path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_predictions_raw(file_path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(f"{row['input']}===>{row['prediction']}\n")


def _save_predictions_raw_by_dataset(
    dataset_name: str,
    file_path: str,
    rows: List[Dict],
    datasets_root: str,
    split: str,
) -> None:
    if dataset_name == "camera-coqe":
        _save_predictions_raw_camera(file_path, rows)
    elif dataset_name == "vcom-data":
        _save_predictions_raw_vcom_mapped(file_path, rows, datasets_root=datasets_root, split=split)
    else:
        _save_predictions_raw(file_path, rows)


def _save_metrics(file_path: str, metrics: Dict[str, Dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)


def _save_metrics_csv(file_path: str, metrics: Dict[str, Dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as fp:
        for line in metrics_to_lines(metrics):
            fp.write(line + "\n")


def main():
    args = parse_args()

    datasets_root = _default_path(args.datasets_root, "datasets")
    output_dir = _default_path(args.output_dir, "llm_eval/results")
    cache_dir = _default_path(args.cache_dir, "llm_eval/cache")

    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]

    summary_rows = []

    for dataset_name in dataset_names:
        samples = load_dataset(dataset_name, datasets_root, args.split)
        if args.limit > 0:
            samples = samples[: args.limit]

        print(f"Dataset={dataset_name} split={args.split} samples={len(samples)}", flush=True)

        for model_name in args.models:
            print(f"  Evaluating model: {model_name}", flush=True)
            model_slug = _slugify(model_name)
            cache_file = os.path.join(cache_dir, dataset_name, args.split, f"{model_slug}.jsonl")
            cache = _load_cache(cache_file)

            if args.provider == "openrouter":
                client = OpenAICompatibleClient(
                    model=model_name,
                    base_url=args.base_url,
                    api_key_env=args.api_key_env,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
            else:
                client = HuggingFaceLocalClient(
                    model=model_name,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                    dtype=args.hf_dtype,
                    load_in_4bit=args.hf_load_in_4bit,
                )

            predictions = []
            gold_labels = []
            prediction_rows = []

            # Fast path for local HF inference: batch uncached requests.
            use_hf_batch = args.provider == "hf-local" and args.inference_batch_size > 1

            if use_hf_batch:
                all_messages: List[List[Dict[str, str]]] = []
                for sample in samples:
                    sentence = sample["input"]
                    language = sample.get("language", "auto")
                    all_messages.append(
                        build_messages(
                            sentence,
                            language=language,
                            dataset=dataset_name,
                            strategy=args.prompt_strategy,
                        )
                    )

                pred_by_idx: Dict[int, str] = {}
                uncached_indices: List[int] = []
                uncached_messages: List[List[Dict[str, str]]] = []

                for idx, sample in enumerate(samples):
                    sentence = sample["input"]
                    if sentence in cache:
                        pred_by_idx[idx] = cache[sentence]
                    else:
                        uncached_indices.append(idx)
                        uncached_messages.append(all_messages[idx])

                for start in tqdm(
                    range(0, len(uncached_indices), args.inference_batch_size),
                    desc=f"{dataset_name} | {model_slug}",
                ):
                    chunk_idx = uncached_indices[start : start + args.inference_batch_size]
                    chunk_msgs = uncached_messages[start : start + args.inference_batch_size]
                    chunk_preds = client.generate_batch(chunk_msgs)
                    for local_i, global_i in enumerate(chunk_idx):
                        pred = chunk_preds[local_i]
                        pred_by_idx[global_i] = pred
                        _append_cache(cache_file, samples[global_i]["input"], pred)

                for sample_idx, sample in enumerate(samples):
                    sentence = sample["input"]
                    gold = sample["output"]
                    pred = pred_by_idx[sample_idx]
                    messages = all_messages[sample_idx]

                    if args.debug_samples > 0 and sample_idx < args.debug_samples:
                        sep = "-" * 60
                        prompt_text = "\n".join(f"[{m['role'].upper()}] {m['content']}" for m in messages)
                        print(f"\n{sep}", flush=True)
                        print(f"[DEBUG sample {sample_idx + 1}/{args.debug_samples}]", flush=True)
                        print(prompt_text, flush=True)
                        print(f"[OUTPUT] {pred}", flush=True)
                        print(sep, flush=True)

                    metric_pred = pred
                    if args.match_mode == "index-match":
                        if dataset_name == "vcom-data":
                            metric_pred = _prediction_to_indexed_vcom(
                                pred,
                                sentence,
                                sample.get("tokenized_input", sentence),
                            )
                        elif dataset_name == "camera-coqe":
                            metric_pred = _prediction_to_indexed_camera(pred, sentence)

                    predictions.append(metric_pred)
                    gold_labels.append(gold)
                    prediction_rows.append(
                        {
                            "dataset": dataset_name,
                            "split": args.split,
                            "model": model_name,
                            "input": sentence,
                            "tokenized_input": sample.get("tokenized_input", sentence),
                            "gold": gold,
                            "prediction": pred,
                        }
                    )
            else:
                for sample_idx, sample in enumerate(tqdm(samples, desc=f"{dataset_name} | {model_slug}")):
                    sentence = sample["input"]
                    gold = sample["output"]
                    language = sample.get("language", "auto")

                    messages = build_messages(
                        sentence,
                        language=language,
                        dataset=dataset_name,
                        strategy=args.prompt_strategy,
                    )
                    if sentence in cache:
                        pred = cache[sentence]
                    else:
                        pred = client.generate(messages)
                        _append_cache(cache_file, sentence, pred)

                    if args.debug_samples > 0 and sample_idx < args.debug_samples:
                        sep = "-" * 60
                        prompt_text = "\n".join(f"[{m['role'].upper()}] {m['content']}" for m in messages)
                        print(f"\n{sep}", flush=True)
                        print(f"[DEBUG sample {sample_idx + 1}/{args.debug_samples}]", flush=True)
                        print(prompt_text, flush=True)
                        print(f"[OUTPUT] {pred}", flush=True)
                        print(sep, flush=True)

                    metric_pred = pred
                    if args.match_mode == "index-match":
                        if dataset_name == "vcom-data":
                            metric_pred = _prediction_to_indexed_vcom(
                                pred,
                                sentence,
                                sample.get("tokenized_input", sentence),
                            )
                        elif dataset_name == "camera-coqe":
                            metric_pred = _prediction_to_indexed_camera(pred, sentence)

                    predictions.append(metric_pred)
                    gold_labels.append(gold)
                    prediction_rows.append(
                        {
                            "dataset": dataset_name,
                            "split": args.split,
                            "model": model_name,
                            "input": sentence,
                            "tokenized_input": sample.get("tokenized_input", sentence),
                            "gold": gold,
                            "prediction": pred,
                        }
                    )

                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)

            _DATASET_LABEL_ORDER = {
                "camera-coqe": CAMERA_COQE_LABEL_ORDER,
                "t5-camera-coqe-data": CAMERA_COQE_LABEL_ORDER,
                "vcom-data": VCOM_LABEL_ORDER,
            }
            label_order = _DATASET_LABEL_ORDER.get(dataset_name)
            if len(predictions) != len(gold_labels):
                raise ValueError(
                    f"Sample count mismatch before evaluation: predictions={len(predictions)} "
                    f"gold_labels={len(gold_labels)} dataset={dataset_name} model={model_name}"
                )
            metrics = compute_coqe_metrics(
                predictions,
                gold_labels,
                label_order=label_order,
                match_mode=args.match_mode,
            )

            pred_file = os.path.join(output_dir, dataset_name, args.split, f"predictions__{model_slug}.jsonl")
            pred_raw_file = os.path.join(output_dir, dataset_name, args.split, f"predictions__{model_slug}.txt")
            metrics_json = os.path.join(output_dir, dataset_name, args.split, f"metrics__{model_slug}.json")
            metrics_csv = os.path.join(output_dir, dataset_name, args.split, f"metrics__{model_slug}.csv")

            _save_predictions(pred_file, prediction_rows)
            _save_predictions_raw_by_dataset(
                dataset_name,
                pred_raw_file,
                prediction_rows,
                datasets_root=datasets_root,
                split=args.split,
            )
            _save_metrics(metrics_json, metrics)
            _save_metrics_csv(metrics_csv, metrics)

            lb = leaderboard_row(metrics)
            ranking_f1 = metrics.get("E-T5-MACRO", {}).get("F1", 0.0)
            e_t4_f1    = metrics.get("E-T4",       {}).get("F1", 0.0)
            e_cee_micro_f1 = metrics.get("E-CEE-MICRO", {}).get("F1", 0.0)
            sent_cmp_f1 = metrics.get("SENT-CMP", {}).get("F1", 0.0)

            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "split": args.split,
                    "model": model_name,
                    "E-T5-MACRO-F1": ranking_f1,
                    "E-T4-F1": e_t4_f1,
                    "E-CEE-MICRO-F1": e_cee_micro_f1,
                    "SENT-CMP-F1": sent_cmp_f1,
                    "match_mode": args.match_mode,
                    "leaderboard": lb,
                }
            )

            print(
                f"    E-T5-MACRO-F1={ranking_f1:.4f}  "
                f"E-T4-F1={e_t4_f1:.4f}  "
                f"E-CEE-MICRO-F1={e_cee_micro_f1:.4f}  "
                f"SENT-CMP-F1={sent_cmp_f1:.4f}",
                flush=True,
            )

    summary_file = os.path.join(output_dir, f"summary__{args.split}.json")
    with open(summary_file, "w", encoding="utf-8") as fp:
        json.dump(summary_rows, fp, ensure_ascii=False, indent=2)

    print("Done. Summary:", summary_file, flush=True)


if __name__ == "__main__":
    main()
