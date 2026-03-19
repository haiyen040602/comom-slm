import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_utils import read_data_file
from dataset import CausalLMDataset
from training import train
from inference import infer
from config import args, data_paths, result_dir, inference_dir, MODEL_NAME, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, MAX_SEQ_LENGTH, NUM_EPOCHS, LEARNING_RATE

if __name__ == "__main__":
    # Load tokenizer and model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if use_mixed_precision else torch.float32,
        device_map="auto",
        use_cache=False  # Ensure compatibility with gradient checkpointing
    )
    tokenizer.pad_token = tokenizer.eos_token

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
    train(
        model, tokenizer, train_dataset, dev_dataset,
        epochs=NUM_EPOCHS, lr=LEARNING_RATE,
        train_batch_size=TRAIN_BATCH_SIZE, eval_batch_size=EVAL_BATCH_SIZE
    )

    # Evaluate the model
    print("Evaluating the model...")
    eval_loss, _, _, _ = infer(dev_dataset, model, tokenizer, batch_size=EVAL_BATCH_SIZE, name="eval")
    print(f"Evaluation Loss: {eval_loss}")