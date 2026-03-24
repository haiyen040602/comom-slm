import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_utils import read_data_file
from dataset import CausalLMDataset, OUTPUT_END_MARKER, build_prompt
from classifier import (
    ClassifierConfig,
    train_comparison_classifier,
    build_comparison_labels,
    compute_binary_metrics,
    print_binary_metrics,
)
from training import train
from inference import infer, generate_predictions_with_comparison_gate, compute_coqe_metrics, print_metrics_table
from logger import TrainingLogger
from config import (args, data_paths, result_dir, inference_dir,
                    MODEL_NAME, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, MAX_SEQ_LENGTH,
                    NUM_EPOCHS, LEARNING_RATE, GRADIENT_ACCUMULATION_STEPS,
                    WEIGHT_DECAY, ADAM_EPSILON, NUM_WARMUP_STEPS,
                    USE_MIXED_PRECISION, USE_GRADIENT_CHECKPOINTING, PROMPT_STYLE,
                    USE_COMPARISON_CLASSIFIER, COMPARISON_MODEL_NAME,
                    COMPARISON_NUM_EPOCHS, COMPARISON_BATCH_SIZE, COMPARISON_LEARNING_RATE)

if __name__ == "__main__":
    # Load tokenizer and model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if USE_MIXED_PRECISION else torch.float32,
        device_map="auto",
        use_cache=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Đảm bảo left-padding cho generation

    # Teach a dedicated output-end marker to improve generation stopping.
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": [OUTPUT_END_MARKER]})
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    # Load data
    print("Loading data...")

    print(f"Train file path: {data_paths['train_file']}")
    print(f"Train file exists: {os.path.exists(data_paths['train_file'])}")
        
    train_inputs, train_labels = read_data_file(data_paths["train_file"])
    dev_inputs, dev_labels = read_data_file(data_paths["dev_file"])

    print(f"Number of training samples: {len(train_inputs)}")
    print(f"Sample training input: {train_inputs[:2]}")
    print(f"Sample training labels: {train_labels[:2]}")

    comparison_model = None
    comparison_tokenizer = None
    train_inputs_causal = train_inputs
    train_labels_causal = train_labels
    if USE_COMPARISON_CLASSIFIER:
        clf_cfg = ClassifierConfig(
            model_name=COMPARISON_MODEL_NAME,
            epochs=COMPARISON_NUM_EPOCHS,
            batch_size=COMPARISON_BATCH_SIZE,
            learning_rate=COMPARISON_LEARNING_RATE,
            max_length=MAX_SEQ_LENGTH,
        )
        comparison_model, comparison_tokenizer = train_comparison_classifier(
            train_inputs,
            train_labels,
            dev_inputs,
            dev_labels,
            clf_cfg,
        )

        # To reduce causal-LM training time, train it only on comparative sentences.
        train_comp_gold = build_comparison_labels(train_labels)
        keep_indices = [i for i, y in enumerate(train_comp_gold) if y == 1]
        if keep_indices:
            train_inputs_causal = [train_inputs[i] for i in keep_indices]
            train_labels_causal = [train_labels[i] for i in keep_indices]
        else:
            print("Warning: no comparative samples found; fallback to full training set.")
        print(f"Causal training samples after comparison filter: {len(train_inputs_causal)}/{len(train_inputs)}")

    train_dataset = CausalLMDataset(
        tokenizer,
        train_inputs_causal,
        train_labels_causal,
        max_len=MAX_SEQ_LENGTH,
        prompt_style=PROMPT_STYLE,
    )
    dev_dataset = CausalLMDataset(tokenizer, dev_inputs, dev_labels, max_len=MAX_SEQ_LENGTH, prompt_style=PROMPT_STYLE)

    # Train the model
    print("Starting training...")
    logger = TrainingLogger(
        log_dir=inference_dir,
        model_name=MODEL_NAME,
        extra_config={
            "prompt_style": PROMPT_STYLE,
            "use_comparison_classifier": USE_COMPARISON_CLASSIFIER,
            "comparison_model_name": COMPARISON_MODEL_NAME if USE_COMPARISON_CLASSIFIER else "",
        },
    )
    logger.log_training_prompts(
        inputs=train_inputs_causal,
        gold_labels=train_labels_causal,
        prompts=[build_prompt(inp, prompt_style=PROMPT_STYLE) for inp in train_inputs_causal],
        formatted_targets=train_dataset.formatted_targets,
        output_end_marker=OUTPUT_END_MARKER,
        max_samples=100,
    )
    train(
        model, tokenizer, train_dataset, dev_dataset,
        epochs=NUM_EPOCHS, lr=LEARNING_RATE,
        train_batch_size=TRAIN_BATCH_SIZE, eval_batch_size=EVAL_BATCH_SIZE,
        acc_step=GRADIENT_ACCUMULATION_STEPS,
        logger=logger,
        comparison_model=comparison_model,
        comparison_tokenizer=comparison_tokenizer,
        comparison_batch_size=COMPARISON_BATCH_SIZE,
    )

    # Evaluate và log kết quả trên tập test
    print("\nRunning inference on test set...")
    test_inputs, test_labels = read_data_file(data_paths["test_file"])
    test_predictions, test_gold, test_traces, test_comp_preds = generate_predictions_with_comparison_gate(
        model=model,
        tokenizer=tokenizer,
        inputs=test_inputs,
        gold_labels=test_labels,
        max_len=MAX_SEQ_LENGTH,
        prompt_style=PROMPT_STYLE,
        eval_batch_size=EVAL_BATCH_SIZE,
        comparison_model=comparison_model,
        comparison_tokenizer=comparison_tokenizer,
        comparison_batch_size=COMPARISON_BATCH_SIZE,
    )

    if comparison_model is not None and comparison_tokenizer is not None:
        test_gold_comp = build_comparison_labels(test_labels)
        test_comp_metrics = compute_binary_metrics(test_comp_preds, test_gold_comp)
        print_binary_metrics(test_comp_metrics, title="Comparison Classifier - Test Gate")

    test_metrics = compute_coqe_metrics(test_predictions, test_gold)
    print_metrics_table(test_metrics, epoch=None)
    logger.log_predictions(
        inputs=test_inputs,
        predictions=test_predictions,
        gold_labels=test_gold,
        metrics=test_metrics,
        split="test"
    )
    logger.log_full_generations(
        traces=test_traces,
        gold_labels=test_gold,
        split="test"
    )