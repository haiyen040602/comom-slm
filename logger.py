"""
logger.py - Ghi log kết quả training, dev evaluation và test predictions
"""
import os
import json
from datetime import datetime


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


class TrainingLogger:
    def __init__(self, log_dir, model_name, extra_config=None):
        self.log_dir   = log_dir
        self.model_name = model_name
        timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id    = f"{model_name.replace('/', '_')}_{timestamp}"

        self.train_log_path  = os.path.join(log_dir, "train_log.jsonl")
        self.dev_log_path    = os.path.join(log_dir, "dev_log.jsonl")
        self.metrics_path    = os.path.join(log_dir, "metrics_summary.json")
        self.test_pred_path  = os.path.join(log_dir, "test_predictions.txt")
        self.dev_pred_path   = os.path.join(log_dir, "dev_predictions.txt")
        self.dev_full_gen_path = os.path.join(log_dir, "dev_full_generations.txt")
        self.test_full_gen_path = os.path.join(log_dir, "test_full_generations.txt")
        self.train_prompt_path = os.path.join(log_dir, "train_prompts_sample.txt")

        os.makedirs(log_dir, exist_ok=True)

        # Ghi header config
        config_path = os.path.join(log_dir, "run_config.json")
        config_payload = {"run_id": self.run_id, "model": model_name, "started_at": timestamp}
        if extra_config:
            config_payload.update(extra_config)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2, ensure_ascii=False)

        self.all_train_epochs = []
        self.all_dev_epochs   = []

        print(f"📁 Logs will be saved to: {log_dir}")

    # ------------------------------------------------------------------
    # Training loss per epoch
    # ------------------------------------------------------------------
    def log_train_epoch(self, epoch, train_loss, eval_loss, lr, metrics: dict):
        """Ghi log sau mỗi epoch: loss + P/R/F1 trên dev."""
        record = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "eval_loss":  round(eval_loss,  6),
            "lr":         lr,
            "metrics":    {k: {m: round(v, 4) for m, v in s.items()}
                           for k, s in metrics.items()},
        }
        self.all_train_epochs.append(record)
        with open(self.train_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Dev / Test predictions
    # ------------------------------------------------------------------
    def log_predictions(self, inputs, predictions, gold_labels, metrics, split="dev"):
        """Ghi file predictions dạng: input ===> gold ||| predicted"""
        path = self.dev_pred_path if split == "dev" else self.test_pred_path

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Run: {self.run_id}  |  Split: {split}\n")
            f.write(f"# Format: input ===> gold ||| predicted\n")
            f.write("-" * 100 + "\n")
            for inp, pred, gold in zip(inputs, predictions, gold_labels):
                f.write(f"{inp} ===> {gold} ||| {pred}\n")

            f.write("\n" + "=" * 100 + "\n")
            f.write("# METRICS\n")
            for name, s in metrics.items():
                support = int(s.get('support', 0))
                f.write(
                    f"  {name:<22} P={s['P']:.4f}  R={s['R']:.4f}  "
                    f"F1={s['F1']:.4f}  support={support}\n"
                )

        print(f"✅ {split.upper()} predictions saved → {path}")

    def log_full_generations(self, traces, gold_labels, split="dev", epoch=None):
        """Save full prompt/raw generation text for debugging model behavior."""
        path = self.dev_full_gen_path if split == "dev" else self.test_full_gen_path
        title = f"# Run: {self.run_id}  |  Split: {split}"
        if epoch is not None:
            title += f"  |  Epoch: {epoch}"

        with open(path, "w", encoding="utf-8") as f:
            f.write(title + "\n")
            f.write("# Format: input + prompt + raw_generated + normalized_prediction + gold\n")
            f.write("=" * 100 + "\n")

            for i, (trace, gold) in enumerate(zip(traces, gold_labels), start=1):
                f.write(f"[{i}] INPUT\n{trace.get('input', '')}\n")
                f.write(f"[{i}] PROMPT\n{trace.get('prompt', '')}\n")
                f.write(f"[{i}] RAW_GENERATED\n{trace.get('raw_generated', '')}\n")
                f.write(f"[{i}] FULL_DECODED\n{trace.get('full_decoded', '')}\n")
                f.write(f"[{i}] NORMALIZED_PREDICTION\n{trace.get('normalized_prediction', '')}\n")
                f.write(f"[{i}] GOLD\n{gold}\n")
                f.write("-" * 100 + "\n")

        print(f"✅ {split.upper()} full generations saved → {path}")

    def log_training_prompts(self, inputs, gold_labels, prompts, formatted_targets=None, output_end_marker="", max_samples=100):
        """Save a sample of the exact prompts and gold outputs used during training."""
        with open(self.train_prompt_path, "w", encoding="utf-8") as f:
            f.write(f"# Run: {self.run_id}  |  Split: train\n")
            f.write(f"# Format: input + prompt + gold + full_train_text  |  Samples: {min(len(inputs), max_samples)}\n")
            f.write("=" * 100 + "\n")

            for i, (inp, gold, prompt) in enumerate(zip(inputs[:max_samples], gold_labels[:max_samples], prompts[:max_samples]), start=1):
                target_for_model = formatted_targets[i-1] if formatted_targets is not None else gold
                full_train_text = f"{prompt} {target_for_model}".strip()
                if output_end_marker:
                    full_train_text = f"{full_train_text} {output_end_marker}".strip()

                f.write(f"[{i}] INPUT\n{inp}\n")
                f.write(f"[{i}] PROMPT\n{prompt}\n")
                f.write(f"[{i}] GOLD\n{gold}\n")
                if formatted_targets is not None:
                    f.write(f"[{i}] TARGET_FOR_MODEL\n{target_for_model}\n")
                f.write(f"[{i}] FULL_TRAIN_TEXT\n{full_train_text}\n")
                f.write("-" * 100 + "\n")

        print(f"✅ TRAIN prompts saved → {self.train_prompt_path}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    def save_summary(self):
        """Lưu file JSON tổng kết toàn bộ quá trình train."""
        summary = {
            "run_id":       self.run_id,
            "model":        self.model_name,
            "num_epochs":   len(self.all_train_epochs),
            "best_epoch":   min(self.all_train_epochs,
                               key=lambda x: x["eval_loss"])["epoch"]
                            if self.all_train_epochs else None,
            "best_eval_loss": min(r["eval_loss"] for r in self.all_train_epochs)
                              if self.all_train_epochs else None,
            "epochs":       self.all_train_epochs,
        }
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✅ Metrics summary saved → {self.metrics_path}")
