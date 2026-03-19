#!/bin/bash

# Check if dataset name is provided
if [ -z "$1" ]; then
  echo "Error: No dataset name provided. Usage: ./setup_and_run.sh <dataset_name>"
  exit 1
fi

DATASET_NAME=$1

# Update and install required packages
pip install -r requirements.txt

# Run the main training and evaluation script
python main.py --data_dir /kaggle/working/datasets \
               --dataset_name $DATASET_NAME \
               --output_dir /kaggle/working \
               --model_name Qwen/Qwen2-0.5B \
               --train_batch_size 8 \
               --eval_batch_size 16 \
               --max_seq_length 256 \
               --num_epochs 3 \
               --learning_rate 5e-5 \
               --gradient_accumulation_steps 4 \
               --weight_decay 0.01 \
               --adam_epsilon 1e-8 \
               --num_warmup_steps 0 \
               --use_mixed_precision True \
               --use_gradient_checkpointing True