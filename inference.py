import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import re

def extract_elements(input_string):
    """Extract structured elements from prediction."""
    input_list = input_string.split(';')
    pattern = re.compile(r'<sub>(.*?)<obj>(.*?)<asp>(.*?)<pred>(.*?)<lab>(.*?)$')
    result = []
    for i in input_list:
        i = i.strip()
        match = re.match(pattern, i[1:-1].strip())
        
        if match:
            items = match.groups()
            new_items = [item.strip() for item in items]
            result.append(new_items)
        else:
            result.append(None)
    
    return result

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

def compute_metrics(predicted_list, gold_list):
    """Compute precision, recall, F1 scores for individual entities and groups."""
    predicted_positions = list(map(list, zip(*predicted_list)))
    gold_positions = list(map(list, zip(*gold_list)))

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for predicted, gold in zip(predicted_positions, gold_positions):
        precision, recall, f1, _ = precision_recall_fscore_support(gold, predicted, average='micro')
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    # Compute metrics for the entire tuple (S, O, A, P, L)
    precision, recall, f1, _ = precision_recall_fscore_support(gold_list, predicted_list, average='micro')
    overall_metrics = {
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

    return precision_scores, recall_scores, f1_scores, overall_metrics

def evaluate_predictions(predicted_list, gold_list, elem_dict):
    """Evaluate predictions against gold labels."""
    assert len(predicted_list) == len(gold_list)

    all_labels, all_predictions = [], []
    for pred, gold in zip(predicted_list, gold_list):
        pred_elements = extract_elements(pred)
        gold_elements = extract_elements(gold)

        for pred_elem, gold_elem in zip(pred_elements, gold_elements):
            if pred_elem and gold_elem:
                all_labels.append(gold_elem)
                all_predictions.append(pred_elem)

    precision_scores, recall_scores, f1_scores, overall_metrics = compute_metrics(all_predictions, all_labels)

    # Print metrics for each entity
    scores_dict = {}
    for i, elem in enumerate(elem_dict):
        scores_dict[elem] = {
            "Precision": precision_scores[i],
            "Recall": recall_scores[i],
            "F1": f1_scores[i]
        }

    print("Entity-wise Metrics:")
    for entity, metrics in scores_dict.items():
        print(f"{entity}: {metrics}")

    print("\nOverall Metrics (S, O, A, P, L):")
    print(overall_metrics)

    return scores_dict, overall_metrics