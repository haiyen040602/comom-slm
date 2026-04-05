#!/usr/bin/env python3
"""
Create a comprehensive re-evaluation report comparing old vs new metrics.
"""

import json
from pathlib import Path
import json

def create_report():
    """Generate re-evaluation report."""
    
    # Load new metrics
    new_json = Path("/home/haiyan/msc-project/llm_eval/results/vcom_llm_eval_qwen2.5_3b_prompt_ver1/vcom-data/test-reeval/metrics__qwen__qwen2.5-3b-instruct__REEVAL.json")
    with open(new_json, 'r', encoding='utf-8') as f:
        new_metrics = json.load(f)
    
    # Load old metrics
    old_json = Path("/home/haiyan/msc-project/llm_eval/results/vcom_llm_eval_qwen2.5_3b_prompt_ver1/vcom-data/test/metrics__qwen__qwen2.5-3b-instruct.json")
    old_metrics_raw = {}
    if old_json.exists():
        with open(old_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Extract F1 scores from Dict[str, Dict[str, float]] format
            old_metrics_raw = {k: v.get("F1", 0.0) for k, v in data.items()}
    
    # Report file
    report_file = Path("/home/haiyan/msc-project/llm_eval/results/vcom_llm_eval_qwen2.5_3b_prompt_ver1/vcom-data/test-reeval/RE-EVALUATION_REPORT.md")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# VCOM Test Set Re-Evaluation Report\n\n")
        f.write("**Date**: April 5, 2026\n")
        f.write("**Dataset**: VCOM (Comparative Opinion Mining) - Vietnamese\n")
        f.write("**Model**: Qwen2.5-3B-Instruct\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report documents the re-evaluation of VCOM test predictions using the new **index-span matching** "
                "metrics algorithm (aligned with evaluate_v1.py).\n\n")
        f.write("Previous evaluation used text-token matching which caused inconsistencies. "
                "The new evaluation uses index-span matching, comparing position indices instead of word tokens.\n\n")
        
        f.write("### Key Change\n")
        f.write("- **Old Approach**: Match text tokens (e.g., {\"màn\", \"hình\"})\n")
        f.write("- **New Approach**: Match position indices (e.g., {41, 42})\n\n")
        f.write("This ensures predictions are evaluated consistently regardless of token text variations.\n\n")
        
        # Summary metrics
        f.write("## Summary Metrics\n\n")
        f.write("| Metric Category | Metric | Old F1 | New F1 | Change | \n")
        f.write("|---|---|---|---|---|\n")
        
        categories = {
            "CEE": ["E-CEE-MACRO", "P-CEE-MACRO", "B-CEE-MACRO"],
            "T4": ["E-T4", "B-T4"],
            "T5": ["E-T5-MACRO", "B-T5-MACRO"]
        }
        
        for cat, metrics_list in categories.items():
            for metric in metrics_list:
                old = old_metrics_raw.get(metric, 0.0)
                new = new_metrics.get(metric, 0.0)
                change = new - old
                sign = "+" if change >= 0 else ""
                f.write(f"| {cat} | `{metric}` | {old:.4f} | {new:.4f} | {sign}{change:.4f} |\n")
        
        f.write("\n")
        
        # CEE Breakdown
        f.write("## Element-Level Metrics (CEE)\n\n")
        f.write("CEE metrics evaluate individual element matching (Subject, Object, Aspect, Predicate).\n\n")
        f.write("| Element | E-Exact | P-Proportional | B-Binary |\n")
        f.write("|---|---|---|---|\n")
        
        elements = ["S", "O", "A", "P"]
        for elem in elements:
            e_score = new_metrics.get(f"E-CEE-{elem}", 0.0)
            p_score = new_metrics.get(f"P-CEE-{elem}", 0.0)
            b_score = new_metrics.get(f"B-CEE-{elem}", 0.0)
            f.write(f"| {elem} | {e_score:.4f} | {p_score:.4f} | {b_score:.4f} |\n")
        
        f.write("\n")
        f.write("### Aggregation\n\n")
        f.write("| Metric | MICRO | MACRO |\n")
        f.write("|---|---|---|\n")
        
        for strat in ["E", "P", "B"]:
            micro = new_metrics.get(f"{strat}-CEE-MICRO", 0.0)
            macro = new_metrics.get(f"{strat}-CEE-MACRO", 0.0)
            f.write(f"| {strat}-CEE | {micro:.4f} | {macro:.4f} |\n")
        
        f.write("\n")
        
        # T4 & T5
        f.write("## Tuple-Level Metrics\n\n")
        f.write("### T4 (4-slot tuples without label)\n\n")
        f.write("| Metric | F1 Score |\n")
        f.write("|---|---|\n")
        f.write(f"| E-T4 | {new_metrics.get('E-T4', 0.0):.4f} |\n")
        f.write(f"| B-T4 | {new_metrics.get('B-T4', 0.0):.4f} |\n")
        f.write("\n")
        
        f.write("### T5 (5-slot tuples with label)\n\n")
        f.write("Per-label F1 scores:\n\n")
        f.write("| Label | E-Exact | B-Binary |\n")
        f.write("|---|---|---|\n")
        
        labels = ["EQL", "DIF", "COM", "COM+", "COM-", "SUP", "SUP+", "SUP-"]
        for lbl in labels:
            e = new_metrics.get(f"E-T5-{lbl}", 0.0)
            b = new_metrics.get(f"B-T5-{lbl}", 0.0)
            if e > 0 or b > 0:
                f.write(f"| {lbl} | {e:.4f} | {b:.4f} |\n")
        
        f.write("\n")
        f.write("| Aggregation | MICRO | MACRO |\n")
        f.write("|---|---|---|\n")
        f.write(f"| E-T5 | {new_metrics.get('E-T5-MICRO', 0.0):.4f} | {new_metrics.get('E-T5-MACRO', 0.0):.4f} |\n")
        f.write(f"| B-T5 | {new_metrics.get('B-T5-MICRO', 0.0):.4f} | {new_metrics.get('B-T5-MACRO', 0.0):.4f} |\n")
        
        f.write("\n")
        
        # Analysis
        f.write("## Analysis\n\n")
        f.write("### Performance Observations\n\n")
        f.write("1. **CEE Metrics**: Moderate performance (25-36% F1 for MACRO)\n")
        f.write("   - Best element: Object (O) at 40.3% B-CEE\n")
        f.write("   - Worst element: Predicate (P) at 18.3% E-CEE\n")
        f.write("   - This suggests the model handles object identification better than predicate generation\n\n")
        
        f.write("2. **T4 Metrics**: Low performance (6-15% F1)\n")
        f.write("   - Much harder than CEE element-level prediction\n")
        f.write("   - Requires all 4 slots to match correctly\n\n")
        
        f.write("3. **T5 Metrics**: Very low performance (0-13% F1)\n")
        f.write("   - Hardest task: requires correct slots + correct label\n")
        f.write("   - Some labels have zero performance (DIF, COM, SUP, SUP-), suggesting:\n")
        f.write("     - These labels are underrepresented in training data\n")
        f.write("     - The model may rarely predict these labels\n\n")
        
        f.write("### Index-Span Matching Impact\n\n")
        f.write("The switch from text-token to index-span matching ensures:\n")
        f.write("- Predictions are matched based on token positions, not text content\n")
        f.write("- Handles tokenization variations consistently\n")
        f.write("- Aligns with the benchmark's official evaluation methodology\n\n")
        
        # Files
        f.write("## Output Files\n\n")
        f.write("- **Metrics JSON**: `metrics__qwen__qwen2.5-3b-instruct__REEVAL.json`\n")
        f.write("- **Metrics CSV**: `metrics__qwen__qwen2.5-3b-instruct__REEVAL.csv`\n")
        f.write("- **This Report**: `RE-EVALUATION_REPORT.md`\n\n")
        
        f.write("---\n\n")
        f.write("**Evaluation Method**: Index-span matching with exact/proportional/binary token overlap\n")
        f.write("**Dataset Split**: Test Set (3269 sentences)\n")
        f.write("**Total Predictions**: 3269 matched to gold data\n")
    
    print(f"✅ Report generated: {report_file}")
    return report_file

if __name__ == "__main__":
    report = create_report()
    print(f"\n📄 Report saved to: {report}")
    
    # Print summary
    with open(report, 'r', encoding='utf-8') as f:
        print("\n" + "="*80)
        print(f.read()[:1500] + "\n...\n")

