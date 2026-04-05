import argparse
import json
import os
from collections import Counter, defaultdict, deque
from typing import Deque, Dict, List, Tuple


def _parse_prediction_blocks(predictions_file: str) -> Dict[str, Deque[List[str]]]:
    blocks: Dict[str, Deque[List[str]]] = defaultdict(deque)

    with open(predictions_file, "r", encoding="utf-8") as fp:
        lines = [line.rstrip("\n") for line in fp]

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue

        if "\t" not in line or line.lstrip().startswith("{"):
            index += 1
            continue

        sentence_line = lines[index]
        sentence = sentence_line.split("\t", 1)[0].strip()
        index += 1

        annotations: List[str] = []
        while index < len(lines):
            current = lines[index].strip()
            if not current:
                index += 1
                break
            if "\t" in current and not current.lstrip().startswith("{"):
                break
            if current.startswith("{"):
                json.loads(current)
                annotations.append(lines[index])
            index += 1

        blocks[sentence].append(annotations)

    return blocks


def _map_predictions_to_raw_files(
    predictions_by_sentence: Dict[str, Deque[List[str]]],
    raw_dir: str,
    output_dir: str,
) -> Tuple[int, int, Counter]:
    os.makedirs(output_dir, exist_ok=True)
    matched_sentences = 0
    total_sentences = 0
    file_annotation_counts: Counter = Counter()

    for file_name in sorted(fn for fn in os.listdir(raw_dir) if fn.endswith(".txt")):
        input_path = os.path.join(raw_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        out_lines: List[str] = []
        with open(input_path, "r", encoding="utf-8") as fp:
            for raw_line in fp:
                line = raw_line.rstrip("\n")
                out_lines.append(line)

                stripped = line.strip()
                if not stripped:
                    continue
                if "\t" not in stripped or stripped.lstrip().startswith("{"):
                    continue

                total_sentences += 1
                sentence = stripped.split("\t", 1)[0].strip()
                queue = predictions_by_sentence.get(sentence)
                if not queue:
                    continue

                annotations = queue.popleft()
                matched_sentences += 1
                for annotation in annotations:
                    out_lines.append(annotation)
                    file_annotation_counts[file_name] += 1

        with open(output_path, "w", encoding="utf-8") as fp:
            fp.write("\n".join(out_lines))
            if out_lines:
                fp.write("\n")

    return matched_sentences, total_sentences, file_annotation_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map vcom prediction blocks from predictions__*.txt back onto raw split files."
    )
    parser.add_argument("--predictions-file", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    predictions_by_sentence = _parse_prediction_blocks(args.predictions_file)
    total_prediction_blocks = sum(len(queue) for queue in predictions_by_sentence.values())

    matched_sentences, total_sentences, file_annotation_counts = _map_predictions_to_raw_files(
        predictions_by_sentence=predictions_by_sentence,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
    )

    remaining_blocks = sum(len(queue) for queue in predictions_by_sentence.values())

    print(f"Loaded prediction blocks: {total_prediction_blocks}")
    print(f"Mapped sentence lines: {matched_sentences}/{total_sentences}")
    print(f"Remaining unmatched prediction blocks: {remaining_blocks}")
    print(f"Files written: {len(file_annotation_counts)} with annotations")


if __name__ == "__main__":
    main()