import re
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


TUPLE_PATTERN = re.compile(
    r'\[S\]\s*(.*?)\s*\[O\]\s*(.*?)\s*\[A\]\s*(.*?)\s*\[P\]\s*(.*?)\s*\[L\]\s*(.*?)(?=\)|\n|;|$)',
    re.DOTALL,
)


@dataclass
class ClassifierConfig:
    model_name: str = "microsoft/deberta-v3-small"
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_length: int = 256


class TextClassificationDataset(Dataset):
    def __init__(self, tokenizer, texts, labels, max_length=256):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _parse_tuples(text):
    tuples = []
    for part in text.split(';'):
        part = part.strip().strip('()')
        match = TUPLE_PATTERN.search(part)
        if match:
            tuples.append(tuple(item.strip() for item in match.groups()))
    return tuples


def _is_all_unk_tuple(t):
    if len(t) < 5:
        return False
    return all((x or '').strip() == '[UNK]' for x in t[:5])


def label_is_comparative(label_text):
    tuples = _parse_tuples(label_text)
    if not tuples:
        return 0
    return 1 if any(not _is_all_unk_tuple(t) for t in tuples) else 0


def build_comparison_labels(label_texts):
    return [label_is_comparative(t) for t in label_texts]


def _safe_div(num, den):
    return num / den if den > 0 else 0.0


def compute_binary_metrics(preds, golds):
    tp = sum(1 for p, g in zip(preds, golds) if p == 1 and g == 1)
    tn = sum(1 for p, g in zip(preds, golds) if p == 0 and g == 0)
    fp = sum(1 for p, g in zip(preds, golds) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(preds, golds) if p == 0 and g == 1)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, len(golds))

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'support_pos': sum(golds),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
    }


def print_binary_metrics(metrics, title='Comparison Classifier'):
    print(f"\n{'='*68}")
    print(f"  {title}")
    print(f"{'='*68}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Support+  : {metrics['support_pos']}")
    print(f"  TP/FP/FN/TN : {metrics['tp']}/{metrics['fp']}/{metrics['fn']}/{metrics['tn']}")
    print(f"{'='*68}\n")


def train_comparison_classifier(train_texts, train_label_texts, dev_texts, dev_label_texts, cfg: ClassifierConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)

    train_labels = build_comparison_labels(train_label_texts)
    dev_labels = build_comparison_labels(dev_label_texts)

    train_dataset = TextClassificationDataset(tokenizer, train_texts, train_labels, max_length=cfg.max_length)
    dev_dataset = TextClassificationDataset(tokenizer, dev_texts, dev_labels, max_length=cfg.max_length)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    print("\n[Stage 1] Training comparison classifier (DeBERTa backbone)...")
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"  Epoch {epoch + 1}/{cfg.epochs} - Train loss: {avg_loss:.4f}")

        dev_preds = predict_comparison_labels(model, tokenizer, dev_texts, batch_size=cfg.batch_size, max_length=cfg.max_length)
        metrics = compute_binary_metrics(dev_preds, dev_labels)
        print_binary_metrics(metrics, title=f"Comparison Classifier - Dev (Epoch {epoch + 1})")

    return model, tokenizer


def predict_comparison_labels(model, tokenizer, texts, batch_size=16, max_length=256):
    device = next(model.parameters()).device
    dataset = TextClassificationDataset(tokenizer, texts, [0] * len(texts), max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            batch_preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
            preds.extend(batch_preds)

    return preds
