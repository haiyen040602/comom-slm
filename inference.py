import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import re
from dataset import OUTPUT_END_MARKER, build_prompt, CausalLMDataset
from classifier import predict_comparison_labels

ELEM_NAMES = ['S', 'O', 'A', 'P', 'L']
ALLOWED_LABELS = {
    'better': 'Better',
    'worse': 'Worse',
    'equal': 'Equal',
    'different': 'Different',
    '[unk]': '[UNK]',
}
TUPLE_INLINE_PATTERN = re.compile(
    r'\[S\]\s*(.*?)\s*\[O\]\s*(.*?)\s*\[A\]\s*(.*?)\s*\[P\]\s*(.*?)\s*\[L\]\s*(.*?)(?=\)|\n|;|$)',
    re.DOTALL,
)
SLOT_PATTERN = re.compile(r'\[(S|O|A|P|L)\]\s*([^\[]*)')

def parse_tuple(text):
    """Parse a single tuple from format: ([S] val [O] val [A] val [P] val [L] val)"""
    text = text.strip().strip('()')
    match = TUPLE_INLINE_PATTERN.search(text)
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
    marker_id = tokenizer.convert_tokens_to_ids(OUTPUT_END_MARKER)
    if marker_id is not None and marker_id != tokenizer.unk_token_id:
        stop_ids.add(marker_id)
    return list(stop_ids)


def generate_dev_predictions(model, tokenizer, dataset, batch_size=16, max_new_tokens=80, return_traces=False):
    """Generate predictions for the dev/test set using the model."""
    model.eval()
    all_predictions = []
    inputs_raw  = dataset.inputs
    targets_raw = dataset.targets
    traces = []
    prompt_style = getattr(dataset, 'prompt_style', 'direct')

    stop_token_ids = _get_stop_token_ids(tokenizer)

    with torch.no_grad():
        for i in tqdm(range(0, len(inputs_raw), batch_size), desc="  Generating", leave=False):
            batch_inputs  = inputs_raw[i:i+batch_size]

            # Encode only the input prompt (no output) for generation
            prompts = [build_prompt(inp, prompt_style=prompt_style) for inp in batch_inputs]
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
                repetition_penalty=1.35,  # phat token lap lai
                no_repeat_ngram_size=4,
            )

            for j, gen_ids in enumerate(generated):
                input_len = encoded['input_ids'].shape[1]
                new_tokens = gen_ids[input_len:]
                raw_generated_text = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
                pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                # Cắt bỏ phần lặp lại sau tuple đầu tiên hợp lệ
                pred_text = _trim_prediction(pred_text)
                all_predictions.append(pred_text)

                if return_traces:
                    prompt_text = prompts[j]
                    mask_row = encoded['attention_mask'][j]
                    non_pad_idx = (mask_row == 1).nonzero(as_tuple=True)[0]
                    start_idx = int(non_pad_idx[0].item()) if len(non_pad_idx) > 0 else 0
                    full_text = tokenizer.decode(gen_ids[start_idx:], skip_special_tokens=False).strip()
                    traces.append({
                        "input": batch_inputs[j],
                        "prompt": prompt_text,
                        "raw_generated": raw_generated_text,
                        "full_decoded": full_text,
                        "normalized_prediction": pred_text,
                    })

    if return_traces:
        return all_predictions, targets_raw, traces
    return all_predictions, targets_raw


def generate_predictions_with_comparison_gate(
    model,
    tokenizer,
    inputs,
    gold_labels,
    max_len=256,
    prompt_style='direct',
    eval_batch_size=16,
    max_new_tokens=80,
    comparison_model=None,
    comparison_tokenizer=None,
    comparison_batch_size=16,
):
    """Run classifier-gated generation.

    - Comparative sentence (pred=1): run causal generation.
    - Non-comparative sentence (pred=0): default prediction is empty string.
    """
    num_samples = len(inputs)
    predictions = [""] * num_samples
    traces = [None] * num_samples

    if comparison_model is not None and comparison_tokenizer is not None:
        comp_preds = predict_comparison_labels(
            comparison_model,
            comparison_tokenizer,
            inputs,
            batch_size=comparison_batch_size,
            max_length=max_len,
        )
    else:
        comp_preds = [1] * num_samples

    comparative_indices = [i for i, pred in enumerate(comp_preds) if pred == 1]

    if comparative_indices:
        comp_inputs = [inputs[i] for i in comparative_indices]
        comp_golds = [gold_labels[i] for i in comparative_indices]
        comp_dataset = CausalLMDataset(
            tokenizer,
            comp_inputs,
            comp_golds,
            max_len=max_len,
            prompt_style=prompt_style,
        )
        comp_predictions, _, comp_traces = generate_dev_predictions(
            model,
            tokenizer,
            comp_dataset,
            batch_size=eval_batch_size,
            max_new_tokens=max_new_tokens,
            return_traces=True,
        )

        for local_idx, global_idx in enumerate(comparative_indices):
            predictions[global_idx] = comp_predictions[local_idx]
            traces[global_idx] = comp_traces[local_idx]

    # Fill traces for non-comparative sentences.
    for i in range(num_samples):
        if traces[i] is None:
            traces[i] = {
                "input": inputs[i],
                "prompt": build_prompt(inputs[i], prompt_style=prompt_style),
                "raw_generated": "",
                "full_decoded": "",
                "normalized_prediction": "",
            }

    return predictions, gold_labels, traces, comp_preds


def _trim_prediction(text):
    """Chi giu cac tuple hop le va cat bo phan sinh du thua."""
    if not text:
        return text

    # Cat nhanh theo cac marker hay gap trong output chat.
    for marker in ["<|im_end|>", "<|endoftext|>", OUTPUT_END_MARKER]:
        if marker in text:
            text = text.split(marker, 1)[0].strip()

    text = _extract_result_section(text)

    # Trich xuat cac tuple hop le duoi dang nhan [S][O][A][P][L] va chuan hoa lai.
    tuples = []
    for m in TUPLE_INLINE_PATTERN.finditer(text):
        s, o, a, p, l = (x.strip() for x in m.groups())
        l = _normalize_label(l, p)
        tuples.append(f"([S] {s} [O] {o} [A] {a} [P] {p} [L] {l})")

    if tuples:
        return ' ; '.join(tuples)

    repaired = _repair_partial_tuple(text)
    if repaired is not None:
        return repaired

    # Fallback: neu khong tim thay tuple day du, chi giu dong dau tien.
    return text.split('\n', 1)[0].strip()


def _extract_result_section(text):
    """Prefer the content after the final Result/Ket qua marker for CoT-style outputs."""
    markers = [r'Result\s*:', r'K[eé]t\s*qu[ảa]\s*:']
    last_match = None
    for pattern in markers:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            last_match = match

    if last_match is not None:
        return text[last_match.end():].strip()
    return text


def _repair_partial_tuple(text):
    """Convert partial slot outputs into a canonical 5-slot tuple when possible."""
    slots = {name: '[UNK]' for name in ELEM_NAMES}
    found_any = False

    for slot_name, slot_value in SLOT_PATTERN.findall(text):
        cleaned = slot_value.strip().strip('() ;')
        cleaned = re.split(r'\n|<\|', cleaned, maxsplit=1)[0].strip()
        if cleaned:
            slots[slot_name] = cleaned
            found_any = True

    if not found_any:
        return None

    slots['L'] = _normalize_label(slots['L'], slots['P'])

    return (
        f"([S] {slots['S']} [O] {slots['O']} [A] {slots['A']} "
        f"[P] {slots['P']} [L] {slots['L']})"
    )


def _normalize_label(label_text, predicate_text=''):
    """Restrict labels to Better/Worse/Equal/Different/[UNK]."""
    label = (label_text or '').strip().strip('()[]').lower()
    predicate = (predicate_text or '').strip().lower()

    # Prefer explicit label tokens if they appear in generated text.
    explicit = re.search(r'\b(better|worse|equal|different|unk)\b', label)
    if explicit:
        token = explicit.group(1)
        if token == 'unk':
            return '[UNK]'
        return token.capitalize()

    if label in ALLOWED_LABELS:
        return ALLOWED_LABELS[label]

    combined = f"{label} {predicate}".strip()

    if any(phrase in combined for phrase in ['better', 'best', 'faster', 'greater', 'more accessible', 'sharper', 'improved', 'excellent', 'lighter', 'longer']):
        return 'Better'
    if any(phrase in combined for phrase in ['worse', 'slower', 'less portable', 'falls short', 'not as good', 'poor', 'bad', 'heavier', 'dim', 'not quite']):
        return 'Worse'
    if any(phrase in combined for phrase in ['different', 'differs', 'larger', 'smaller', 'higher', 'lower']):
        return 'Different'
    if any(phrase in combined for phrase in ['no difference', 'same', 'similar', 'equal', 'consistent', 'on par']):
        return 'Equal'

    return '[UNK]'


def _prf(tp, pred, gold):
    p  = tp / pred if pred > 0 else 0.0
    r  = tp / gold if gold > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1, gold


def _is_all_unk_tuple(t):
    """Return True if all 5 slots in tuple are [UNK]."""
    if len(t) < 5:
        return False
    return all((x or '').strip() == '[UNK]' for x in t[:5])


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

        # Exclude all-UNK tuples from tuple-level evaluation because
        # they represent non-comparative sentences.
        gold_cmp_tuples = [t for t in gold_tuples if not _is_all_unk_tuple(t)]
        pred_cmp_tuples = [t for t in pred_tuples if not _is_all_unk_tuple(t)]

        gold_set_4 = [t[:4] for t in gold_cmp_tuples if len(t) >= 4]
        gold_set_5 = list(gold_cmp_tuples)
        pred_set_4 = [t[:4] for t in pred_cmp_tuples if len(t) >= 4]
        pred_set_5 = list(pred_cmp_tuples)

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

        # Per-element: evaluate only comparative tuples.
        # Non-comparative (all-UNK) gold/pred should not become TP.
        for idx, e in enumerate(ELEM_NAMES):
            g_vals = [gt[idx] for gt in gold_cmp_tuples if len(gt) > idx]
            p_vals = [pt[idx] for pt in pred_cmp_tuples if len(pt) > idx]
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
        p, r, f1, support = _prf(elem_tp[e], elem_pred[e], elem_gold[e])
        results[e] = {'P': p, 'R': r, 'F1': f1, 'support': support}

    p4, r4, f14, support4 = _prf(tp_4, pred_4, gold_4)
    p5, r5, f15, support5 = _prf(tp_5, pred_5, gold_5)
    results['4-tuple (S,O,A,P)'] = {'P': p4, 'R': r4, 'F1': f14, 'support': support4}
    results['5-tuple (S,O,A,P,L)'] = {'P': p5, 'R': r5, 'F1': f15, 'support': support5}
    return results


def print_metrics_table(metrics, epoch=None):
    """Print metrics in a formatted table"""
    title = "Dev Metrics" if epoch is None else f"Dev Metrics - Epoch {epoch}"
    print(f"\n{'-'*72}")
    print(f"  {title}")
    print(f"{'-'*72}")
    print(f"  {'Element':<20} {'Precision':>9} {'Recall':>9} {'F1':>9} {'Support':>8}")
    print(f"  {'-'*64}")
    for name, s in metrics.items():
        marker = ''
        print(f"  {name:<20} {s['P']:>9.4f} {s['R']:>9.4f} {s['F1']:>9.4f} {s.get('support', 0):>8d}{marker}")
    print(f"{'-'*72}\n")