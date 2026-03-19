import argparse
import os

# Default configuration values
DEFAULT_MODEL_NAME = "Qwen/Qwen2-0.5B"
DEFAULT_TRAIN_BATCH_SIZE = 8
DEFAULT_EVAL_BATCH_SIZE = 16
DEFAULT_MAX_SEQ_LENGTH = 256
DEFAULT_NUM_EPOCHS = 3
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_ADAM_EPSILON = 1e-8
DEFAULT_NUM_WARMUP_STEPS = 0
DEFAULT_USE_MIXED_PRECISION = True
DEFAULT_USE_GRADIENT_CHECKPOINTING = True
DEFAULT_SPECIAL_TOKENS = ['[S]', '[O]', '[A]', '[P]', '[L]', '[UNK]', '(', ')', ';', ':', 'Better', 'Worse', 'Equal', 'Different']

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for training and evaluation")
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME, help="Model name or path")
    parser.add_argument('--train_batch_size', type=int, default=DEFAULT_TRAIN_BATCH_SIZE, help="Training batch size")
    parser.add_argument('--eval_batch_size', type=int, default=DEFAULT_EVAL_BATCH_SIZE, help="Evaluation batch size")
    parser.add_argument('--max_seq_length', type=int, default=DEFAULT_MAX_SEQ_LENGTH, help="Maximum sequence length")
    parser.add_argument('--num_epochs', type=int, default=DEFAULT_NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS, help="Gradient accumulation steps")
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay for optimizer")
    parser.add_argument('--adam_epsilon', type=float, default=DEFAULT_ADAM_EPSILON, help="Epsilon for Adam optimizer")
    parser.add_argument('--num_warmup_steps', type=int, default=DEFAULT_NUM_WARMUP_STEPS, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument('--use_mixed_precision', type=bool, default=DEFAULT_USE_MIXED_PRECISION, help="Use mixed precision training")
    parser.add_argument('--use_gradient_checkpointing', type=bool, default=DEFAULT_USE_GRADIENT_CHECKPOINTING, help="Use gradient checkpointing")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory")
    return parser.parse_args()

# Get configuration from command-line arguments
args = parse_args()

# Assign configuration values
MODEL_NAME = args.model_name
TRAIN_BATCH_SIZE = args.train_batch_size
EVAL_BATCH_SIZE = args.eval_batch_size
MAX_SEQ_LENGTH = args.max_seq_length
NUM_EPOCHS = args.num_epochs
LEARNING_RATE = args.learning_rate
GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
WEIGHT_DECAY = args.weight_decay
ADAM_EPSILON = args.adam_epsilon
NUM_WARMUP_STEPS = args.num_warmup_steps
USE_MIXED_PRECISION = args.use_mixed_precision
USE_GRADIENT_CHECKPOINTING = args.use_gradient_checkpointing
SPECIAL_TOKENS = DEFAULT_SPECIAL_TOKENS

# Data paths and output directories
data_paths = {
    "train_file": os.path.join(args.data_dir, "train.txt"),
    "dev_file": os.path.join(args.data_dir, "dev.txt"),
    "test_file": os.path.join(args.data_dir, "test.txt")
}

result_dir = os.path.join(args.output_dir, f"result/model-{MODEL_NAME}")
inference_dir = os.path.join(args.output_dir, f"result/inference-{MODEL_NAME}")
os.makedirs(result_dir, exist_ok=True)
os.makedirs(inference_dir, exist_ok=True)