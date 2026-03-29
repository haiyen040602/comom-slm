# LLM Evaluation for Comparative Opinion Mining

This folder contains a standalone evaluation pipeline to benchmark popular LLMs on:
- t5-camera-coqe-data (English)
- vcom-data (Vietnamese)

The evaluator normalizes both datasets into the same COQE tuple format and reports:
- S, O, A, P, L precision/recall/F1
- 4-tuple (S,O,A,P)
- 5-tuple (S,O,A,P,L)

## 1) Install dependencies

```bash
cd llm_eval
pip install -r requirements.txt
```

## 2) Set API key

Recommended: OpenRouter so you can evaluate many popular models with one API.

```bash
export OPENROUTER_API_KEY="your_key_here"
```

If you use another OpenAI-compatible endpoint, set its API key env and pass --api-key-env.

## 3) Run evaluation

Example on test split, both datasets, with multiple popular models:

```bash
python run_eval.py \
  --datasets t5-camera-coqe-data,vcom-data \
  --split test \
  --models openai/gpt-4o-mini anthropic/claude-3.5-sonnet google/gemini-2.0-flash-001 deepseek/deepseek-chat \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY
```

Quick smoke test with first 50 samples:

```bash
python run_eval.py \
  --datasets t5-camera-coqe-data,vcom-data \
  --split test \
  --models openai/gpt-4o-mini \
  --limit 50
```

## 4) Output files

Generated under llm_eval/results:
- per dataset/split/model predictions JSONL
- per dataset/split/model metrics JSON and CSV
- summary file: summary__<split>.json

Cache is stored under llm_eval/cache to avoid repeated API calls.

## Notes

- The script uses strict tuple output prompting.
- For vcom-data, labels are mapped to COQE labels:
  - COM+, SUP+ -> Better
  - COM-, SUP- -> Worse
  - EQL -> Equal
  - DIF -> Different
- Sentences without comparative annotation are mapped to [UNK] tuple.
