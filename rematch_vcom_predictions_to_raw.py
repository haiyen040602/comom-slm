#!/usr/bin/env python3
"""Rematch VCOM predictions to raw test order and fill missing rows with empty labels."""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

PRED_TUPLE_RE = re.compile(
    r"\[S\]\s*(.*?)\s*\[O\]\s*(.*?)\s*\[A\]\s*(.*?)\s*\[P\]\s*(.*?)\s*\[L\]\s*(.*?)(?=\)|\n|;|$)",
    re.DOTALL,
)


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def is_unk_or_empty(text: str) -> bool:
    t = (text or "").strip()
    return t == "" or t == "[UNK]"


def parse_pred_tuples(text: str) -> List[Dict[str, str]]:
    tuples: List[Dict[str, str]] = []
    for part in (text or "").split(";"):
        m = PRED_TUPLE_RE.search(part.strip().strip("()"))
        if not m:
            continue
        s, o, a, p, l = (x.strip() for x in m.groups())
        tuples.append({"S": s, "O": o, "A": a, "P": p, "L": l})
    return tuples


def span_positions_from_tokenized(slot_value: str, tokenized_tokens: List[str]) -> List[int]:
    slot_value = (slot_value or "").strip()
    if not slot_value or not tokenized_tokens:
        return []

    slot_tokens = slot_value.split()
    m = len(slot_tokens)

    # Exact contiguous
    for i in range(len(tokenized_tokens) - m + 1):
        if tokenized_tokens[i : i + m] == slot_tokens:
            return list(range(i + 1, i + m + 1))

    # Greedy non-contiguous fallback
    positions = []
    cursor = 0
    for token in slot_tokens:
        found = -1
        for j in range(cursor, len(tokenized_tokens)):
            if tokenized_tokens[j] == token:
                found = j
                break
        if found < 0:
            for j in range(0, len(tokenized_tokens)):
                if tokenized_tokens[j] == token:
                    found = j
                    break
        if found < 0:
            return []
        positions.append(found + 1)
        cursor = found + 1
    return positions


def to_indexed_slot(slot_value: str, tokenized_tokens: List[str]) -> List[str]:
    if is_unk_or_empty(slot_value):
        return []
    positions = span_positions_from_tokenized(slot_value, tokenized_tokens)
    if not positions:
        return []
    slot_tokens = slot_value.split()
    if len(slot_tokens) != len(positions):
        n = min(len(slot_tokens), len(positions))
        slot_tokens = slot_tokens[:n]
        positions = positions[:n]
    return [f"{pos}&&{tok}" for pos, tok in zip(positions, slot_tokens)]


def prediction_to_vcom_json_lines(pred: str, tokenized_sentence: str) -> List[str]:
    tuples = parse_pred_tuples(pred)
    tokenized_tokens = tokenized_sentence.split()
    out_lines: List[str] = []

    for t in tuples:
        if all(is_unk_or_empty(t[k]) for k in ("S", "O", "A", "P", "L")):
            continue

        obj = {
            "subject": to_indexed_slot(t["S"], tokenized_tokens),
            "object": to_indexed_slot(t["O"], tokenized_tokens),
            "aspect": to_indexed_slot(t["A"], tokenized_tokens),
            "predicate": to_indexed_slot(t["P"], tokenized_tokens),
            "label": t["L"] if not is_unk_or_empty(t["L"]) else "",
        }
        out_lines.append(json.dumps(obj, ensure_ascii=False))

    return out_lines


def load_raw_sentences(raw_test_dir: Path) -> List[Tuple[str, str, str]]:
    """Return list of (file_name, sentence, tokenized_sentence) in raw order."""
    rows: List[Tuple[str, str, str]] = []
    for fp in sorted(raw_test_dir.glob("test_*.txt")):
        for line in fp.read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if not t:
                continue
            if "\t" in t and not t.startswith("{"):
                parts = t.split("\t", 1)
                sent = parts[0].strip()
                tok = parts[1].strip() if len(parts) > 1 else sent
                rows.append((fp.name, sent, tok))
    return rows


def load_prediction_map(pred_jsonl: Path) -> DefaultDict[str, List[Dict[str, str]]]:
    """Map normalized sentence -> queue of prediction rows (to handle duplicates)."""
    pred_map: DefaultDict[str, List[Dict[str, str]]] = defaultdict(list)
    for line in pred_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = normalize_text(row.get("input", ""))
        pred_map[key].append(row)
    return pred_map


def main() -> None:
    root = Path("/home/haiyan/msc-project")
    raw_test_dir = root / "datasets" / "vcom-data" / "test"
    pred_jsonl = (
        root
        / "llm_eval"
        / "results"
        / "vcom_llm_eval_qwen2.5_3b_prompt_ver1"
        / "vcom-data"
        / "test"
        / "predictions__qwen__qwen2.5-3b-instruct.jsonl"
    )

    out_dir = (
        root
        / "llm_eval"
        / "results"
        / "vcom_llm_eval_qwen2.5_3b_prompt_ver1"
        / "vcom-data"
        / "test-reeval"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "predictions__qwen__qwen2.5-3b-instruct__MATCHED_RAW_WITH_EMPTY_LABEL.txt"
    per_file_dir = out_dir / "test_map_with_empty_label"
    per_file_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = load_raw_sentences(raw_test_dir)
    pred_map = load_prediction_map(pred_jsonl)

    total = len(raw_rows)
    matched = 0
    missing = 0
    extra = 0

    out_lines: List[str] = []
    per_file_lines: DefaultDict[str, List[str]] = defaultdict(list)

    for file_name, sentence, tokenized_sentence in raw_rows:
        header_line = f"{sentence}\t{tokenized_sentence}"
        out_lines.append(header_line)
        per_file_lines[file_name].append(header_line)

        key = normalize_text(sentence)
        if pred_map[key]:
            row = pred_map[key].pop(0)
            pred_lines = prediction_to_vcom_json_lines(row.get("prediction", ""), tokenized_sentence)
            if pred_lines:
                out_lines.extend(pred_lines)
                per_file_lines[file_name].extend(pred_lines)
            else:
                # Prediction exists but produced no tuples -> keep empty-label row.
                empty_json = json.dumps(
                    {
                        "subject": [],
                        "object": [],
                        "aspect": [],
                        "predicate": [],
                        "label": "",
                    },
                    ensure_ascii=False,
                )
                out_lines.append(empty_json)
                per_file_lines[file_name].append(empty_json)
            matched += 1
        else:
            empty_json = json.dumps(
                {
                    "subject": [],
                    "object": [],
                    "aspect": [],
                    "predicate": [],
                    "label": "",
                },
                ensure_ascii=False,
            )
            out_lines.append(empty_json)
            per_file_lines[file_name].append(empty_json)
            missing += 1

        out_lines.append("")
        per_file_lines[file_name].append("")

    for key in pred_map:
        extra += len(pred_map[key])

    out_file.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")

    for file_name, lines in per_file_lines.items():
        target = per_file_dir / file_name
        target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    summary = {
        "raw_sentence_count": total,
        "matched_prediction_count": matched,
        "missing_filled_with_empty_label": missing,
        "unused_prediction_rows": extra,
        "output_file": str(out_file),
        "output_per_file_dir": str(per_file_dir),
    }

    summary_file = out_dir / "predictions_match_summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
