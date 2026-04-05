#!/usr/bin/env python3
"""
Re-evaluate VCOM predictions using the new index-span matching metrics.
Loads existing predictions from results/vcom_llm_eval_qwen2.5_3b_prompt_ver1/test_map/
and re-computes metrics with the updated metrics.py (index-span based).
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Add llm_eval to path
sys.path.insert(0, '/home/haiyan/msc-project')

from llm_eval.metrics import compute_coqe_metrics, metrics_to_lines, VCOM_LABEL_ORDER


def load_vcom_predictions_from_test_map() -> Tuple[List[str], List[str], List[str]]:
    """Load predictions from individual test_map files."""
    test_map_dir = Path("/home/haiyan/msc-project/llm_eval/results/vcom_llm_eval_qwen2.5_3b_prompt_ver1/test_map")
    
    pred_sentences = []
    pred_tuples = []
    pred_tokenized = []
    
    # Get all test files sorted
    test_files = sorted(test_map_dir.glob("test_*.txt"))
    print(f"📁 Found {len(test_files)} test prediction files in test_map/")
    
    for test_file in test_files:
        with open(test_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Parse format: sentence | tokenized | JSON tuples
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Check if this is a sentence line (contains tab)
            if '\t' in line:
                parts = line.split('\t')
                sent = parts[0].strip()
                tokenized = parts[1].strip() if len(parts) > 1 else sent
                
                # Collect all JSON tuples following this sentence
                tuples_list = []
                i += 1
                while i < len(lines):
                    json_line = lines[i].strip()
                    if not json_line:
                        i += 1
                        break
                    if json_line.startswith('{'):
                        tuples_list.append(json_line)
                        i += 1
                    else:
                        break
                
                # Add this sample
                pred_sentences.append(sent)
                pred_tokenized.append(tokenized)
                pred_tuples.append(tuples_list)
            else:
                i += 1
    
    print(f"✅ Loaded {len(pred_sentences)} sentences with predictions")
    return pred_sentences, pred_tokenized, pred_tuples


def load_vcom_gold_data() -> Tuple[List[str], List[str], List[str]]:
    """Load gold data from consolidated test.txt."""
    test_file = Path("/home/haiyan/msc-project/datasets/vcom-data/test.txt")
    
    gold_sentences = []
    gold_tokenized = []
    gold_tuples = []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Check if this is a sentence line (contains tab)
        if '\t' in line:
            parts = line.split('\t')
            sent = parts[0].strip()
            tokenized = parts[1].strip() if len(parts) > 1 else sent
            
            # Collect all JSON tuples
            tuples_list = []
            i += 1
            while i < len(lines):
                json_line = lines[i].strip()
                if not json_line:
                    i += 1
                    break
                if json_line.startswith('{'):
                    tuples_list.append(json_line)
                    i += 1
                else:
                    break
            
            gold_sentences.append(sent)
            gold_tokenized.append(tokenized)
            gold_tuples.append(tuples_list)
        else:
            i += 1
    
    print(f"✅ Loaded {len(gold_sentences)} sentences from gold data")
    return gold_sentences, gold_tokenized, gold_tuples


def convert_tuples_to_format(tuples_list: List[str]) -> str:
    """Convert JSON tuples to COQE format string."""
    if not tuples_list:
        return "([S] [UNK] [O] [UNK] [A] [UNK] [P] [UNK] [L] [UNK])"
    
    # Parse all JSON tuples and convert to COQE format
    coqe_tuples = []
    for json_str in tuples_list:
        try:
            data = json.loads(json_str)
            subject = " ".join(data.get("subject", []))
            obj = " ".join(data.get("object", []))
            aspect = " ".join(data.get("aspect", []))
            predicate = " ".join(data.get("predicate", []))
            label = data.get("label", "UNK")
            
            coqe_tuple = f"([S] {subject or '[UNK]'} [O] {obj or '[UNK]'} [A] {aspect or '[UNK]'} [P] {predicate or '[UNK]'} [L] {label})"
            coqe_tuples.append(coqe_tuple)
        except json.JSONDecodeError:
            continue
    
    if not coqe_tuples:
        return "([S] [UNK] [O] [UNK] [A] [UNK] [P] [UNK] [L] [UNK])"
    
    return " ".join(coqe_tuples)


def align_predictions_to_gold(pred_sents, pred_tokenized, pred_tuples, 
                              gold_sents, gold_tokenized, gold_tuples) -> Tuple[List[str], List[str]]:
    """
    Align predictions to gold by matching sentences.
    Returns aligned (pred_outputs, gold_outputs) for metric computation.
    """
    pred_outputs = []
    gold_outputs = []
    
    matched = 0
    unmatched = 0
    
    for i, pred_sent in enumerate(pred_sents):
        # Find matching gold sentence
        found = False
        for j, gold_sent in enumerate(gold_sents):
            if pred_sent.lower() == gold_sent.lower():
                # Convert tuples to COQE format
                pred_output = convert_tuples_to_format(pred_tuples[i])
                gold_output = convert_tuples_to_format(gold_tuples[j])
                
                pred_outputs.append(pred_output)
                gold_outputs.append(gold_output)
                matched += 1
                found = True
                break
        
        if not found:
            unmatched += 1
    
    print(f"⚙️  Aligned {matched}/{len(pred_sents)} predictions to gold")
    if unmatched > 0:
        print(f"⚠️  {unmatched} predictions could not be matched")
    
    return pred_outputs, gold_outputs


def main():
    print("🔄 Re-evaluating VCOM predictions with new index-span matching...\n")
    
    # Load predictions
    print("📥 Loading predictions...")
    pred_sents, pred_tokenized, pred_tuples = load_vcom_predictions_from_test_map()
    
    # Load gold data
    print("📥 Loading gold data...")
    gold_sents, gold_tokenized, gold_tuples = load_vcom_gold_data()
    
    # Align predictions to gold
    print("🔗 Aligning predictions to gold...")
    pred_outputs, gold_outputs = align_predictions_to_gold(
        pred_sents, pred_tokenized, pred_tuples,
        gold_sents, gold_tokenized, gold_tuples
    )
    
    # Compute metrics with new index-span matching
    print("\n⏱️  Computing metrics with index-span matching...")
    metrics = compute_coqe_metrics(pred_outputs, gold_outputs, label_order=VCOM_LABEL_ORDER)
    
    # Create output directory
    output_dir = Path("/home/haiyan/msc-project/llm_eval/results/vcom_llm_eval_qwen2.5_3b_prompt_ver1/vcom-data/test-reeval")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Flatten metrics for JSON save (metric-name -> F1 score)
    metrics_flat = {k: v.get("F1", 0.0) for k, v in metrics.items()}
    
    # Save metrics as JSON
    json_file = output_dir / "metrics__qwen__qwen2.5-3b-instruct__REEVAL.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_flat, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved metrics to {json_file}")
    
    # Save metrics as CSV
    csv_file = output_dir / "metrics__qwen__qwen2.5-3b-instruct__REEVAL.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        for line in metrics_to_lines(metrics):
            f.write(line + '\n')
    print(f"✅ Saved metrics CSV to {csv_file}")
    
    # Load old metrics for comparison
    old_json_file = Path("/home/haiyan/msc-project/llm_eval/results/vcom_llm_eval_qwen2.5_3b_prompt_ver1/vcom-data/test/metrics__qwen__qwen2.5-3b-instruct.json")
    if old_json_file.exists():
        print("\n📊 Comparing with old metrics...")
        with open(old_json_file, 'r', encoding='utf-8') as f:
            old_metrics_data = json.load(f)
        
        # Handle both formats: Dict[str, float] or Dict[str, Dict[str, float]]
        if old_metrics_data and isinstance(list(old_metrics_data.values())[0], dict):
            # It's Dict[str, Dict[str, float]]
            old_metrics_flat = {k: v.get("F1", 0.0) for k, v in old_metrics_data.items()}
        else:
            # It's Dict[str, float]
            old_metrics_flat = old_metrics_data
        
        # Compare key metrics
        key_metrics = [
            "E-CEE-MACRO", "P-CEE-MACRO", "B-CEE-MACRO",
            "E-T5-MACRO", "B-T5-MACRO"
        ]
        
        print("\n📈 Key Metric Comparison:")
        print(f"{'Metric':<25} {'Old Value':<15} {'New Value':<15} {'Difference':<15}")
        print("-" * 70)
        
        for metric in key_metrics:
            old_val = old_metrics_flat.get(metric, 0)
            new_val = metrics_flat.get(metric, 0)
            diff = new_val - old_val
            
            print(f"{metric:<25} {old_val:<15.6f} {new_val:<15.6f} {diff:+.6f}")
    
    print("\n✨ Re-evaluation complete!")


if __name__ == "__main__":
    main()
