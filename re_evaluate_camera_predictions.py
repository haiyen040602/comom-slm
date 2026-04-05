#!/usr/bin/env python3
"""Re-match camera-coqe prediction indices and re-evaluate using current logic."""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from llm_eval.metrics import compute_coqe_metrics, metrics_to_lines, CAMERA_COQE_LABEL_ORDER

PRED_TUPLE_RE = re.compile(
    r"\[S\]\s*(.*?)\s*\[O\]\s*(.*?)\s*\[A\]\s*(.*?)\s*\[P\]\s*(.*?)\s*\[L\]\s*(.*?)(?=\)|\n|;|$)",
    re.DOTALL,
)
WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
PAIR_RE = re.compile(r"\[(.*?)\]")

CAMERA_LABEL_MAP = {
    "-1": "Worse",
    "0": "Equal",
    "1": "Better",
    "2": "Different",
}

CAMERA_LABEL_INV = {
    "Worse": "-1",
    "Equal": "0",
    "Better": "1",
    "Different": "2",
    "[UNK]": "",
    "": "",
}


def parse_pred_tuples(text: str) -> List[Dict[str, str]]:
    tuples: List[Dict[str, str]] = []
    for part in (text or "").split(";"):
        m = PRED_TUPLE_RE.search(part.strip().strip("()"))
        if not m:
            continue
        s, o, a, p, l = (x.strip() for x in m.groups())
        tuples.append({"S": s, "O": o, "A": a, "P": p, "L": l})
    return tuples


def is_unk_or_empty(text: str) -> bool:
    t = (text or "").strip()
    return t == "" or t == "[UNK]"


def tokenize_with_offsets(text: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for m in WORD_RE.finditer(text):
        tok = m.group(0)
        out.append({
            "raw": tok,
            "norm": tok.lower(),
            "start": m.start(),
            "end": m.end(),
        })
    return out


def span_token_positions(sentence: str, span_text: str) -> List[int]:
    span_text = (span_text or "").strip()
    if is_unk_or_empty(span_text):
        return []

    sent_tokens = tokenize_with_offsets(sentence)
    span_tokens = tokenize_with_offsets(span_text)
    if not sent_tokens or not span_tokens:
        return []

    sent_norm = [t["norm"] for t in sent_tokens]
    span_norm = [t["norm"] for t in span_tokens]
    m = len(span_norm)

    for i in range(0, len(sent_norm) - m + 1):
        if sent_norm[i : i + m] == span_norm:
            return list(range(i + 1, i + m + 1))

    idx = sentence.lower().find(span_text.lower())
    if idx >= 0:
        end = idx + len(span_text)
        positions = []
        for i, tok in enumerate(sent_tokens, start=1):
            if not (tok["end"] <= idx or tok["start"] >= end):
                positions.append(i)
        if positions:
            return positions

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


def to_indexed_slot(slot_value: str, sentence: str) -> str:
    if is_unk_or_empty(slot_value):
        return ""
    positions = span_token_positions(sentence, slot_value)
    if not positions:
        return ""
    slot_tokens = tokenize_with_offsets(slot_value)
    if len(slot_tokens) != len(positions):
        n = min(len(slot_tokens), len(positions))
        slot_tokens = slot_tokens[:n]
        positions = positions[:n]
    return " ".join(f"{pos}&&{tok['raw']}" for pos, tok in zip(positions, slot_tokens))


def pred_to_indexed_canonical(pred: str, sentence: str) -> str:
    tuples = parse_pred_tuples(pred)
    out_parts: List[str] = []
    for t in tuples:
        if all(is_unk_or_empty(t[k]) for k in ("S", "O", "A", "P", "L")):
            continue
        s = to_indexed_slot(t["S"], sentence) or "[UNK]"
        o = to_indexed_slot(t["O"], sentence) or "[UNK]"
        a = to_indexed_slot(t["A"], sentence) or "[UNK]"
        p = to_indexed_slot(t["P"], sentence) or "[UNK]"
        l = t["L"] if not is_unk_or_empty(t["L"]) else "[UNK]"
        out_parts.append(f"([S] {s} [O] {o} [A] {a} [P] {p} [L] {l})")
    if out_parts:
        return " ; ".join(out_parts)
    return "([S] [UNK] [O] [UNK] [A] [UNK] [P] [UNK] [L] [UNK])"


def canonical_to_raw_lines(indexed_pred: str, sentence: str) -> List[str]:
    tuples = parse_pred_tuples(indexed_pred)
    valid = []
    for t in tuples:
        if all(is_unk_or_empty(t[k]) for k in ("S", "O", "A", "P", "L")):
            continue
        valid.append(t)

    flag = "1" if valid else "0"
    out = [f"{sentence}\t{flag}"]
    if not valid:
        out.append("[[];[];[];[];[]]")
        return out

    for t in valid:
        s = "" if t["S"] == "[UNK]" else t["S"]
        o = "" if t["O"] == "[UNK]" else t["O"]
        a = "" if t["A"] == "[UNK]" else t["A"]
        p = "" if t["P"] == "[UNK]" else t["P"]
        l = CAMERA_LABEL_INV.get(t["L"], "")
        out.append(f"[[{s}];[{o}];[{a}];[{p}];[{l}]]")
    return out


def parse_gold_camera_raw(data_file: Path) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    cur_sentence = ""
    cur_flag = "0"
    cur_anns: List[str] = []

    def flush() -> None:
        nonlocal cur_sentence, cur_flag, cur_anns
        if not cur_sentence:
            return
        if cur_flag == "1" and cur_anns:
            output = " ; ".join(cur_anns)
        else:
            output = "([S] [UNK] [O] [UNK] [A] [UNK] [P] [UNK] [L] [UNK])"
        samples.append({"input": cur_sentence, "gold": output})
        cur_sentence = ""
        cur_flag = "0"
        cur_anns = []

    for line in data_file.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t:
            continue

        if "\t" in t and not t.startswith("["):
            flush()
            s, f = t.rsplit("\t", 1)
            cur_sentence = s.strip()
            cur_flag = f.strip()
            continue

        if t.startswith("[") and cur_sentence:
            pairs = PAIR_RE.findall(t)
            if len(pairs) < 5:
                continue
            s = pairs[0].strip() or "[UNK]"
            o = pairs[1].strip() or "[UNK]"
            a = pairs[2].strip() or "[UNK]"
            p = pairs[3].strip() or "[UNK]"
            label = CAMERA_LABEL_MAP.get(pairs[4].strip(), "[UNK]")
            cur_anns.append(f"([S] {s} [O] {o} [A] {a} [P] {p} [L] {label})")

    flush()
    return samples


def flatten_metrics(metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {k: v.get("F1", 0.0) for k, v in metrics.items()}


def process_result_folder(result_dir: Path, gold_samples: List[Dict[str, object]]) -> Tuple[str, Dict[str, float]]:
    pred_jsonl_files = sorted((result_dir / "camera-coqe" / "test").glob("predictions__*.jsonl"))
    if not pred_jsonl_files:
        raise FileNotFoundError(f"No predictions jsonl in {result_dir}")

    pred_jsonl = pred_jsonl_files[0]
    rows = [json.loads(x) for x in pred_jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]

    if len(rows) != len(gold_samples):
        raise ValueError(f"Row count mismatch in {pred_jsonl}: pred={len(rows)}, gold={len(gold_samples)}")

    pred_indexed: List[str] = []
    gold_indexed: List[str] = []
    raw_lines: List[str] = []

    for i, row in enumerate(rows):
        sentence = row.get("input", "")
        if sentence != gold_samples[i]["input"]:
            raise ValueError(
                f"Sentence mismatch at {i} in {pred_jsonl}\n"
                f"pred: {sentence}\n"
                f"gold: {gold_samples[i]['input']}"
            )

        indexed = pred_to_indexed_canonical(row.get("prediction", ""), sentence)
        pred_indexed.append(indexed)
        gold_indexed.append(gold_samples[i]["gold"])

        raw_lines.extend(canonical_to_raw_lines(indexed, sentence))

    metrics = compute_coqe_metrics(pred_indexed, gold_indexed, label_order=CAMERA_COQE_LABEL_ORDER)
    metrics_flat = flatten_metrics(metrics)

    out_dir = result_dir / "camera-coqe" / "test-reeval-index"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_tag = pred_jsonl.stem.replace("predictions", "")

    out_raw = out_dir / f"predictions_rematch_index{model_tag}.txt"
    out_json = out_dir / f"metrics_rematch_index{model_tag}.json"
    out_csv = out_dir / f"metrics_rematch_index{model_tag}.csv"

    out_raw.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(metrics_flat, f, ensure_ascii=False, indent=2)
    with out_csv.open("w", encoding="utf-8") as f:
        for line in metrics_to_lines(metrics):
            f.write(line + "\n")

    return result_dir.name, metrics_flat


def main() -> None:
    root = Path("/home/haiyan/msc-project")
    results_root = root / "llm_eval" / "results"
    gold_file = root / "datasets" / "camera-coqe" / "test.txt"

    gold_samples = parse_gold_camera_raw(gold_file)
    print(f"Loaded gold camera-coqe samples: {len(gold_samples)}")

    targets = sorted(results_root.glob("cameracoqe-*"))
    if not targets:
        raise FileNotFoundError("No cameracoqe-* result folders found")

    summary = {}
    for folder in targets:
        name, metrics_flat = process_result_folder(folder, gold_samples)
        summary[name] = {
            "E-CEE-MACRO": metrics_flat.get("E-CEE-MACRO", 0.0),
            "P-CEE-MACRO": metrics_flat.get("P-CEE-MACRO", 0.0),
            "B-CEE-MACRO": metrics_flat.get("B-CEE-MACRO", 0.0),
            "E-T4": metrics_flat.get("E-T4", 0.0),
            "B-T4": metrics_flat.get("B-T4", 0.0),
            "E-T5-MACRO": metrics_flat.get("E-T5-MACRO", 0.0),
            "B-T5-MACRO": metrics_flat.get("B-T5-MACRO", 0.0),
        }
        print(f"Done: {name}")

    summary_file = results_root / "camera_coqe_reeval_index_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nRe-evaluation summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary: {summary_file}")


if __name__ == "__main__":
    main()
