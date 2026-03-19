import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_utils import read_data_file
from dataset import CausalLMDataset
from training import train
from inference import infer, generate_dev_predictions, compute_coqe_metrics, print_metrics_table
from logger import TrainingLogger
from config import (args, data_paths, result_dir, inference_dir,
                    MODEL_NAME, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, MAX_SEQ_LENGTH,
                    NUM_EPOCHS, LEARNING_RATE, GRADIENT_ACCUMULATION_STEPS,
                    WEIGHT_DECAY, ADAM_EPSILON, NUM_WARMUP_STEPS,
                    USE_MIXED_PRECISION, USE_GRADIENT_CHECKPOINTING)

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

    # Load data
    print("Loading data...")

    print(f"Train file path: {data_paths['train_file']}")
    print(f"Train file exists: {os.path.exists(data_paths['train_file'])}")
        
    train_inputs, train_labels = read_data_file(data_paths["train_file"])
    dev_inputs, dev_labels = read_data_file(data_paths["dev_file"])

    print(f"Number of training samples: {len(train_inputs)}")
    print(f"Sample training input: {train_inputs[:2]}")
    print(f"Sample training labels: {train_labels[:2]}")

    train_dataset = CausalLMDataset(tokenizer, train_inputs, train_labels, max_len=MAX_SEQ_LENGTH)
    dev_dataset = CausalLMDataset(tokenizer, dev_inputs, dev_labels, max_len=MAX_SEQ_LENGTH)

    # Train the model
    print("Starting training...")
    logger = TrainingLogger(log_dir=inference_dir, model_name=MODEL_NAME)
    train(
        model, tokenizer, train_dataset, dev_dataset,
        epochs=NUM_EPOCHS, lr=LEARNING_RATE,
        train_batch_size=TRAIN_BATCH_SIZE, eval_batch_size=EVAL_BATCH_SIZE,
        acc_step=GRADIENT_ACCUMULATION_STEPS,
        logger=logger
    )

    # Evaluate và log kết quả trên tập test
    print("\nRunning inference on test set...")
    test_inputs, test_labels = read_data_file(data_paths["test_file"])
    test_dataset = CausalLMDataset(tokenizer, test_inputs, test_labels, max_len=MAX_SEQ_LENGTH)
    test_predictions, test_gold = generate_dev_predictions(
        model, tokenizer, test_dataset, batch_size=EVAL_BATCH_SIZE
    )
    test_metrics = compute_coqe_metrics(test_predictions, test_gold)
    print_metrics_table(test_metrics, epoch=None)
    logger.log_predictions(
        inputs=test_inputs,
        predictions=test_predictions,
        gold_labels=test_gold,
        metrics=test_metrics,
        split="test"
    )