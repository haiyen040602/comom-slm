import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_utils import read_data_file
from dataset import CausalLMDataset
from training import train
from inference import infer

if __name__ == "__main__":
    # Configuration
    model_name = "Qwen/Qwen2-0.5B"
    data_dir = "t5-camera-coqe-data"
    train_file = "train.txt"
    dev_file = "dev.txt"
    test_file = "test.txt"
    
    train_batch_size = 8
    eval_batch_size = 16
    max_seq_length = 256
    num_epochs = 3
    learning_rate = 5e-5

    # Load tokenizer and model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print("Loading data...")
    train_inputs, train_labels = read_data_file(os.path.join(data_dir, train_file))
    dev_inputs, dev_labels = read_data_file(os.path.join(data_dir, dev_file))

    train_dataset = CausalLMDataset(tokenizer, train_inputs, train_labels, max_len=max_seq_length)
    dev_dataset = CausalLMDataset(tokenizer, dev_inputs, dev_labels, max_len=max_seq_length)

    # Train the model
    print("Starting training...")
    train(
        model, tokenizer, train_dataset, dev_dataset,
        epochs=num_epochs, lr=learning_rate,
        train_batch_size=train_batch_size, eval_batch_size=eval_batch_size
    )

    # Evaluate the model
    print("Evaluating the model...")
    eval_loss, _, _, _ = infer(dev_dataset, model, tokenizer, batch_size=eval_batch_size, name="eval")
    print(f"Evaluation Loss: {eval_loss}")