import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import re

ELEM_NAMES = ['S', 'O', 'A', 'P', 'L']

def parse_tuple(text):
    """Parse a single tuple from format: ([S] val [O] val [A] val [P] val [L] val)"""
    text = text.strip().strip('()')
    pattern = re.compile(r'\[S\](.*?)\[O\](.*?)\[A\](.*?)\[P\](.*?)\[L\](.*?)$')
    match = pattern.match(text.strip())
    if match:
        return tuple(item.strip() for item in match.groups())
    return None

def parse_output(text):
    """Parse output string with multiple tuples separated by ;"""
    tuples = []
    for part in text.split(';'):
        t = parse_tuple(part.strip())
        if t:
            tuples.append(t)
    return tuples

def infer(dataset, model, tokenizer, batch_size, max_seq_length=256, name="eval", verbose=False):
    """Inference for causal LM - generates predictions"""
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    
    inputs, outputs, targets = [], [], []
    average_loss = 0
    
    model.eval()
    
    with torch.no_grad():
        if name == "eval":
            # Evaluation: compute loss
            total_loss = 0
            num_batches = len(data_loader)
            
            for batch in tqdm(data_loader, desc="Evaluating", disable=not verbose):
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                labels = batch['labels'].to(model.device)
                
                outputs_batch = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs_batch.loss
                total_loss += loss.item()
            
            average_loss = total_loss / num_batches
        else:
            # Inference: generate outputs
            for batch in tqdm(data_loader, desc=f"Inferencing ({name})", disable=not verbose):
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                
                # Generate
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_seq_length,
                    num_beams=1,
                    early_stopping=True,
                    do_sample=False
                )
                
                # Decode
                for i, gen_ids in enumerate(generated_ids):
                    text = tokenizer.decode(gen_ids, skip_special_tokens=False)
                    input_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    
                    # Extract output part after "Output:"
                    if "Output:" in text:
                        output_text = text.split("Output:")[-1].strip()
                    else:
                        output_text = text
                    
                    inputs.append(input_text)
                    outputs.append(output_text)
    
    return average_loss, inputs, outputs, targets

def _get_stop_token_ids(tokenizer):
    """Lấy danh sách token IDs dùng để dừng generation (EOS + chat end tokens)."""
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    # Qwen chat end-of-turn token
    for token in ["<|im_end|>", "<|endoftext|>", "<eos>"]:
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)
    return list(stop_ids)


def generate_dev_predictions(model, tokenizer, dataset, batch_size=16, max_new_tokens=80):
    """Generate predictions for the dev/test set using the model"""
    model.eval()
    all_predictions = []
    inputs_raw  = dataset.inputs
    targets_raw = dataset.targets

    stop_token_ids = _get_stop_token_ids(tokenizer)

    with torch.no_grad():
        for i in tqdm(range(0, len(inputs_raw), batch_size), desc="  Generating", leave=False):
            batch_inputs  = inputs_raw[i:i+batch_size]

            # Encode only the input prompt (no output) for generation
            prompts = [f"Input: {inp}\nOutput:" for inp in batch_inputs]
            encoded = tokenizer(
                prompts, padding=True, truncation=True,
                max_length=256, return_tensors="pt"
            ).to(model.device)

            generated = model.generate(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
                repetition_penalty=1.3,   # Phạt token lặp lại
            )

            for j, gen_ids in enumerate(generated):
                input_len = encoded['input_ids'].shape[1]
                new_tokens = gen_ids[input_len:]
                pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                # Cắt bỏ phần lặp lại sau tuple đầu tiên hợp lệ
                pred_text = _trim_prediction(pred_text)
                all_predictions.append(pred_text)

    return all_predictions, targets_raw


def _trim_prediction(text):
    """Giữ lại các tuple hợp lệ ([S]...[L]...) và bỏ phần nhiễu phía sau."""
    # Thu thập tất cả các tuple hợp lệ
    parts = text.split(';')
    valid_parts = []
    for part in parts:
        part = part.strip()
        # Dừng ngay khi gặp phần không phải tuple hợp lệ
        if re.search(r'\[S\].*\[O\].*\[A\].*\[P\].*\[L\]', part):
            valid_parts.append(part)
        elif not part:
            continue
        else:
            break  # Phần nhiễu bắt đầu, dừng lại
    return ' ; '.join(valid_parts) if valid_parts else text.split('\n')[0].strip()


def _prf(tp, pred, gold):
    p  = tp / pred if pred > 0 else 0.0
    r  = tp / gold if gold > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def compute_coqe_metrics(predictions, gold_labels):
    """Compute P, R, F1 for each element (S,O,A,P,L), 4-tuple and 5-tuple.

    Matching strategy: greedy set-based matching per sample.
    """
    elem_tp   = {e: 0 for e in ELEM_NAMES}
    elem_pred = {e: 0 for e in ELEM_NAMES}
    elem_gold = {e: 0 for e in ELEM_NAMES}
    tp_4, pred_4, gold_4 = 0, 0, 0
    tp_5, pred_5, gold_5 = 0, 0, 0

    for pred_str, gold_str in zip(predictions, gold_labels):
        pred_tuples = parse_output(pred_str)
        gold_tuples = parse_output(gold_str)

        gold_set_4 = [t[:4] for t in gold_tuples if len(t) >= 4]
        gold_set_5 = list(gold_tuples)
        pred_set_4 = [t[:4] for t in pred_tuples if len(t) >= 4]
        pred_set_5 = list(pred_tuples)

        gold_4 += len(gold_set_4)
        gold_5 += len(gold_set_5)
        pred_4 += len(pred_set_4)
        pred_5 += len(pred_set_5)

        # Greedy 4-tuple match
        used = set()
        for pt4 in pred_set_4:
            for gi, gt4 in enumerate(gold_set_4):
                if gi not in used and pt4 == gt4:
                    tp_4 += 1
                    used.add(gi)
                    break

        # Greedy 5-tuple match
        used = set()
        for pt5 in pred_set_5:
            for gi, gt5 in enumerate(gold_set_5):
                if gi not in used and pt5 == gt5:
                    tp_5 += 1
                    used.add(gi)
                    break

        # Per-element: greedy match per position
        for idx, e in enumerate(ELEM_NAMES):
            g_vals = [gt[idx] for gt in gold_tuples if len(gt) > idx]
            p_vals = [pt[idx] for pt in pred_tuples if len(pt) > idx]
            elem_gold[e] += len(g_vals)
            elem_pred[e] += len(p_vals)
            used = set()
            for pv in p_vals:
                for gi, gv in enumerate(g_vals):
                    if gi not in used and pv == gv:
                        elem_tp[e] += 1
                        used.add(gi)
                        break

    results = {}
    for e in ELEM_NAMES:
        results[e] = dict(zip(['P','R','F1'], _prf(elem_tp[e], elem_pred[e], elem_gold[e])))
    results['4-tuple (S,O,A,P)'] = dict(zip(['P','R','F1'], _prf(tp_4, pred_4, gold_4)))
    results['5-tuple (S,O,A,P,L)'] = dict(zip(['P','R','F1'], _prf(tp_5, pred_5, gold_5)))
    return results


def print_metrics_table(metrics, epoch=None):
    """Print metrics in a formatted table"""
    title = "Dev Metrics" if epoch is None else f"Dev Metrics — Epoch {epoch}"
    print(f"\n{'─'*58}")
    print(f"  {title}")
    print(f"{'─'*58}")
    print(f"  {'Element':<20} {'Precision':>9} {'Recall':>9} {'F1':>9}")
    print(f"  {'─'*50}")
    for name, s in metrics.items():
        marker = ' ◀' if 'tuple' in name else ''
        print(f"  {name:<20} {s['P']:>9.4f} {s['R']:>9.4f} {s['F1']:>9.4f}{marker}")
    print(f"{'─'*58}\n")