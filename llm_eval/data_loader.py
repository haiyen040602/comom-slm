"""Data loaders for COQE datasets.

Each sample is returned as a dict:
    input   : str       — the raw sentence
    output  : str       — semicolon-separated canonical tuples:
                          ([S] ... [O] ... [A] ... [P] ... [L] ...)
    language: str       — 'en' | 'vi'
    dataset : str       — dataset identifier

Label conventions (kept as-is for each dataset):
    t5-camera-coqe-data (EN): Better | Worse | Equal | Different | [UNK]
    camera-coqe raw     (EN): -1 | 0 | 1 | 2  (mapped to Worse|Equal|Better|Different)
    vcom-data           (VI): COM+ | COM- | COM | SUP+ | SUP- | SUP | EQL | DIF
"""

import json
import os
import re
from typing import Dict, List, Optional

# Sentinel tuple used when a sentence has no comparative annotation.
EMPTY_TUPLE = "([S] [UNK] [O] [UNK] [A] [UNK] [P] [UNK] [L] [UNK])"

# All valid label strings per dataset (used for prompt construction & validation).
T5_LABELS   = {"Better", "Worse", "Equal", "Different", "[UNK]"}
VCOM_LABELS = {"COM+", "COM-", "COM", "SUP+", "SUP-", "SUP", "EQL", "DIF"}
CAMERA_COQE_RAW_LABELS = {"-1", "0", "1", "2"}

CAMERA_COQE_LABEL_MAP = {
    "-1": "Worse",
    "0": "Equal",
    "1": "Better",
    "2": "Different",
}


# ── Public API ────────────────────────────────────────────────────────────────

def load_dataset(dataset_name: str, datasets_root: str, split: str) -> List[Dict]:
    """Load a dataset split and return a list of sample dicts."""
    if dataset_name == "t5-camera-coqe-data":
        return _load_t5(datasets_root, split)
    if dataset_name == "camera-coqe":
        return _load_camera_coqe_raw(datasets_root, split)
    if dataset_name == "vcom-data":
        return _load_vcom(datasets_root, split)
    raise ValueError(f"Unsupported dataset: '{dataset_name}'")


# ── t5-camera-coqe-data (English) ────────────────────────────────────────────

def _load_t5(datasets_root: str, split: str) -> List[Dict]:
    data_path = os.path.join(datasets_root, "t5-camera-coqe-data", f"{split}.txt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

    samples: List[Dict] = []
    with open(data_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line or "===>" not in line:
                continue
            parts = line.split("===>", 1)
            sentence = parts[0].strip()
            output   = parts[1].strip()
            if sentence and output:
                samples.append({
                    "input":   sentence,
                    "output":  output,
                    "language": "en",
                    "dataset": "t5-camera-coqe-data",
                })
    return samples


# ── camera-coqe raw (English, sentence + binary flag + bracket tuples) ──────

_CAMERA_COQE_PAIR_RE = re.compile(r"\[(.*?)\]")


def _parse_camera_coqe_slot(slot_text: str) -> str:
    """Parse one camera-coqe slot and preserve index tokens when present.

    Example: '10&&RAW 11&&format' -> '10&&RAW 11&&format'
    Empty slot (e.g., '') is converted to [UNK].
    """
    slot_text = slot_text.strip()
    if not slot_text:
        return "[UNK]"

    parts = [part.strip().strip("[]") for part in slot_text.split() if part.strip()]
    return " ".join(parts) if parts else "[UNK]"


def _parse_camera_coqe_annotation(line: str) -> str:
    """Convert raw annotation format into canonical COQE tuple string.

    Raw format example:
        [[10&&RAW 11&&format];[14&&JPEG 15&&format];[2&&file-size];[5&&bigger];[1]]
    """
    pairs = _CAMERA_COQE_PAIR_RE.findall(line.strip())
    if len(pairs) < 5:
        return EMPTY_TUPLE

    subject = _parse_camera_coqe_slot(pairs[0])
    obj = _parse_camera_coqe_slot(pairs[1])
    aspect = _parse_camera_coqe_slot(pairs[2])
    predicate = _parse_camera_coqe_slot(pairs[3])

    raw_label = pairs[4].strip()
    label = CAMERA_COQE_LABEL_MAP.get(raw_label, "[UNK]")

    return (
        f"([S] {subject} [O] {obj} [A] {aspect} "
        f"[P] {predicate} [L] {label})"
    )


def _load_camera_coqe_raw(datasets_root: str, split: str) -> List[Dict]:
    data_path = os.path.join(datasets_root, "camera-coqe", f"{split}.txt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

    samples: List[Dict] = []
    current_sentence = ""
    current_flag = "0"
    current_tuples: List[str] = []

    def flush_current() -> None:
        nonlocal current_sentence, current_flag, current_tuples
        if not current_sentence:
            return

        if current_flag == "1" and current_tuples:
            output = " ; ".join(current_tuples)
        else:
            output = EMPTY_TUPLE

        samples.append(
            {
                "input": current_sentence,
                "output": output,
                "language": "en",
                "dataset": "camera-coqe",
            }
        )

        current_sentence = ""
        current_flag = "0"
        current_tuples = []

    with open(data_path, "r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue

            # Sentence line: "sentence<TAB>0|1"
            if "\t" in line and not line.startswith("["):
                flush_current()
                parts = line.rsplit("\t", 1)
                if len(parts) == 2:
                    current_sentence = parts[0].strip()
                    current_flag = parts[1].strip()
                else:
                    current_sentence = line.strip()
                    current_flag = "0"
                continue

            # Annotation line
            if line.startswith("[") and current_sentence:
                tuple_str = _parse_camera_coqe_annotation(line)
                if tuple_str != EMPTY_TUPLE:
                    current_tuples.append(tuple_str)

    flush_current()
    return samples


# ── vcom-data (Vietnamese) ────────────────────────────────────────────────────

_METADATA_RE = re.compile(
    r"^(\d+\.?|title\s*:|alt\s*:|des\s*:)",
    re.IGNORECASE,
)


def _is_metadata(sentence: str) -> bool:
    s = sentence.strip()
    return not s or bool(_METADATA_RE.match(s))


def _tokens_to_text(tokens: List[str]) -> str:
    """Convert ['1&&word', '2&&word2'] → 'word word2', falling back to '[UNK]'."""
    words = []
    for tok in tokens:
        part = tok.split("&&", 1)[-1].strip()  # works whether or not '&&' present
        if part:
            words.append(part)
    return " ".join(words) if words else "[UNK]"


def _vcom_annotation_to_tuple(ann: Dict) -> str:
    """Convert a vcom JSON annotation to canonical tuple string.

    Labels are kept verbatim (COM+/COM-/COM/SUP+/SUP-/SUP/EQL/DIF).
    """
    # Keep index-span tokens for evaluate_v1-compatible scoring in llm_eval.metrics.
    subject   = " ".join(ann.get("subject",   [])) or "[UNK]"
    obj       = " ".join(ann.get("object",    [])) or "[UNK]"
    aspect    = " ".join(ann.get("aspect",    [])) or "[UNK]"
    predicate = " ".join(ann.get("predicate", [])) or "[UNK]"
    label     = ann.get("label", "[UNK]").strip() or "[UNK]"
    return f"([S] {subject} [O] {obj} [A] {aspect} [P] {predicate} [L] {label})"


def _flush_vcom(sentence: str, anns: List[Dict], out: List[Dict], tokenized_input: str = "") -> None:
    if not sentence or _is_metadata(sentence):
        return
    if anns:
        output = " ; ".join(_vcom_annotation_to_tuple(a) for a in anns)
    else:
        output = EMPTY_TUPLE
    out.append({
        "input":    sentence,
        "output":   output,
        "language": "vi",
        "dataset":  "vcom-data",
        "tokenized_input": tokenized_input or sentence,  # Use original if not provided
    })


def _load_vcom(datasets_root: str, split: str) -> List[Dict]:
    split_dir = os.path.join(datasets_root, "vcom-data", split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Directory not found: {split_dir}")

    samples: List[Dict] = []
    for file_name in sorted(fn for fn in os.listdir(split_dir) if fn.endswith(".txt")):
        file_path = os.path.join(split_dir, file_name)
        cur_sentence = ""
        cur_tokenized = ""
        cur_anns: List[Dict] = []

        with open(file_path, "r", encoding="utf-8") as fp:
            for raw_line in fp:
                line = raw_line.strip()
                if not line:
                    continue

                # Sentence line: contains a tab, does NOT start with '{'
                if "\t" in line and not line.lstrip().startswith("{"):
                    _flush_vcom(cur_sentence, cur_anns, samples, cur_tokenized)
                    cur_anns = []
                    parts = line.split("\t", 1)
                    cur_sentence = parts[0].strip()
                    cur_tokenized = parts[1].strip() if len(parts) > 1 else cur_sentence
                    continue

                # Annotation JSON
                if line.startswith("{") and cur_sentence:
                    try:
                        ann = json.loads(line)
                        if isinstance(ann, dict) and "label" in ann:
                            cur_anns.append(ann)
                    except json.JSONDecodeError:
                        pass

        _flush_vcom(cur_sentence, cur_anns, samples, cur_tokenized)

    return samples
