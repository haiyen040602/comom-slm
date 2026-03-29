import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List

# Allow running from any working directory (e.g., from a cloud notebook).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tqdm import tqdm

from client import OpenAICompatibleClient
from data_loader import load_dataset
from metrics import compute_coqe_metrics, metrics_to_lines
from prompts import build_messages

from metrics import compute_coqe_metrics, leaderboard_row, metrics_to_lines
from prompts import build_messages

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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="Set >0 for quick smoke tests")
    parser.add_argument("--datasets-root", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--cache-dir", type=str, default="")
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

        print(f"Dataset={dataset_name} split={args.split} samples={len(samples)}")

        for model_name in args.models:
            print(f"  Evaluating model: {model_name}")
            model_slug = _slugify(model_name)
            cache_file = os.path.join(cache_dir, dataset_name, args.split, f"{model_slug}.jsonl")
            cache = _load_cache(cache_file)

            client = OpenAICompatibleClient(
                model=model_name,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
            )

            predictions = []
            gold_labels = []
            prediction_rows = []

            for sample in tqdm(samples, desc=f"{dataset_name} | {model_slug}"):
                sentence = sample["input"]
                gold = sample["output"]
                language = sample.get("language", "auto")

                if sentence in cache:
                    pred = cache[sentence]
                else:
                        messages = build_messages(sentence, language=language, dataset=dataset_name)
                        pred = client.generate(messages)
                        _append_cache(cache_file, sentence, pred)

                predictions.append(pred)
                gold_labels.append(gold)
                prediction_rows.append(
                    {
                        "dataset": dataset_name,
                        "split": args.split,
                        "model": model_name,
                        "input": sentence,
                        "gold": gold,
                        "prediction": pred,
                    }
                )

                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

            metrics = compute_coqe_metrics(predictions, gold_labels)

            pred_file = os.path.join(output_dir, dataset_name, args.split, f"predictions__{model_slug}.jsonl")
            metrics_json = os.path.join(output_dir, dataset_name, args.split, f"metrics__{model_slug}.json")
            metrics_csv = os.path.join(output_dir, dataset_name, args.split, f"metrics__{model_slug}.csv")

            _save_predictions(pred_file, prediction_rows)
            _save_metrics(metrics_json, metrics)
            _save_metrics_csv(metrics_csv, metrics)

            lb = leaderboard_row(metrics)
            ranking_f1 = metrics.get("E-T5-MACRO", {}).get("F1", 0.0)
            e_t4_f1    = metrics.get("E-T4",       {}).get("F1", 0.0)
            e_cee_micro_f1 = metrics.get("E-CEE-MICRO", {}).get("F1", 0.0)

            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "split": args.split,
                    "model": model_name,
                    "E-T5-MACRO-F1": ranking_f1,
                    "E-T4-F1": e_t4_f1,
                    "E-CEE-MICRO-F1": e_cee_micro_f1,
                    "leaderboard": lb,
                }
            )

            print(
                f"    E-T5-MACRO-F1={ranking_f1:.4f}  "
                f"E-T4-F1={e_t4_f1:.4f}  "
                f"E-CEE-MICRO-F1={e_cee_micro_f1:.4f}"
            )

    summary_file = os.path.join(output_dir, f"summary__{args.split}.json")
    with open(summary_file, "w", encoding="utf-8") as fp:
        json.dump(summary_rows, fp, ensure_ascii=False, indent=2)

    print("Done. Summary:", summary_file)


if __name__ == "__main__":
    main()
