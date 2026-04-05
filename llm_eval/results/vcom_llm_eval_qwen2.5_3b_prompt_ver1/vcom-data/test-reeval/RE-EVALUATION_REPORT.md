# VCOM Test Set Re-Evaluation Report

**Date**: April 5, 2026
**Dataset**: VCOM (Comparative Opinion Mining) - Vietnamese
**Model**: Qwen2.5-3B-Instruct

## Overview

This report documents the re-evaluation of VCOM test predictions using the new **index-span matching** metrics algorithm (aligned with evaluate_v1.py).

Previous evaluation used text-token matching which caused inconsistencies. The new evaluation uses index-span matching, comparing position indices instead of word tokens.

### Key Change
- **Old Approach**: Match text tokens (e.g., {"màn", "hình"})
- **New Approach**: Match position indices (e.g., {41, 42})

This ensures predictions are evaluated consistently regardless of token text variations.

## Summary Metrics

| Metric Category | Metric | Old F1 | New F1 | Change | 
|---|---|---|---|---|
| CEE | `E-CEE-MACRO` | 0.0000 | 0.2552 | +0.2552 |
| CEE | `P-CEE-MACRO` | 0.0000 | 0.3337 | +0.3337 |
| CEE | `B-CEE-MACRO` | 0.0000 | 0.3626 | +0.3626 |
| T4 | `E-T4` | 0.0000 | 0.0635 | +0.0635 |
| T4 | `B-T4` | 0.0000 | 0.1500 | +0.1500 |
| T5 | `E-T5-MACRO` | 0.0000 | 0.0190 | +0.0190 |
| T5 | `B-T5-MACRO` | 0.0000 | 0.0533 | +0.0533 |

## Element-Level Metrics (CEE)

CEE metrics evaluate individual element matching (Subject, Object, Aspect, Predicate).

| Element | E-Exact | P-Proportional | B-Binary |
|---|---|---|---|
| S | 0.2828 | 0.3271 | 0.3428 |
| O | 0.3135 | 0.3770 | 0.4028 |
| A | 0.2417 | 0.2992 | 0.3125 |
| P | 0.1830 | 0.3316 | 0.3922 |

### Aggregation

| Metric | MICRO | MACRO |
|---|---|---|
| E-CEE | 0.2493 | 0.2552 |
| P-CEE | 0.3302 | 0.3337 |
| B-CEE | 0.3601 | 0.3626 |

## Tuple-Level Metrics

### T4 (4-slot tuples without label)

| Metric | F1 Score |
|---|---|
| E-T4 | 0.0635 |
| B-T4 | 0.1500 |

### T5 (5-slot tuples with label)

Per-label F1 scores:

| Label | E-Exact | B-Binary |
|---|---|---|
| EQL | 0.0423 | 0.1088 |
| DIF | 0.0000 | 0.0155 |
| COM+ | 0.0571 | 0.1320 |
| COM- | 0.0526 | 0.1009 |
| SUP+ | 0.0000 | 0.0693 |

| Aggregation | MICRO | MACRO |
|---|---|---|
| E-T5 | 0.0440 | 0.0190 |
| B-T5 | 0.1071 | 0.0533 |

## Analysis

### Performance Observations

1. **CEE Metrics**: Moderate performance (25-36% F1 for MACRO)
   - Best element: Object (O) at 40.3% B-CEE
   - Worst element: Predicate (P) at 18.3% E-CEE
   - This suggests the model handles object identification better than predicate generation

2. **T4 Metrics**: Low performance (6-15% F1)
   - Much harder than CEE element-level prediction
   - Requires all 4 slots to match correctly

3. **T5 Metrics**: Very low performance (0-13% F1)
   - Hardest task: requires correct slots + correct label
   - Some labels have zero performance (DIF, COM, SUP, SUP-), suggesting:
     - These labels are underrepresented in training data
     - The model may rarely predict these labels

### Index-Span Matching Impact

The switch from text-token to index-span matching ensures:
- Predictions are matched based on token positions, not text content
- Handles tokenization variations consistently
- Aligns with the benchmark's official evaluation methodology

## Output Files

- **Metrics JSON**: `metrics__qwen__qwen2.5-3b-instruct__REEVAL.json`
- **Metrics CSV**: `metrics__qwen__qwen2.5-3b-instruct__REEVAL.csv`
- **This Report**: `RE-EVALUATION_REPORT.md`

---

**Evaluation Method**: Index-span matching with exact/proportional/binary token overlap
**Dataset Split**: Test Set (3269 sentences)
**Total Predictions**: 3269 matched to gold data
