import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup
from torch.amp import GradScaler, autocast
from inference import generate_dev_predictions, compute_coqe_metrics, print_metrics_table

def train(model, tokenizer, train_data, val_data, epochs, lr, train_batch_size, eval_batch_size, acc_step=4):
    """Training function for Causal LM with memory optimization"""
    print("#" * 20 + " BEGIN TRAINING " + "#" * 20)
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    print("✓ Gradient checkpointing enabled")
    
    # Print sample data for verification
    print_sample_data(train_data, tokenizer)
    
    # Set model to training mode
    model.train()
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=eval_batch_size)

    num_training_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    scaler = GradScaler()

    best_eval_loss = float('inf')
    train_losses, eval_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        step_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", ncols=120)
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            with autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / acc_step

            scaler.scale(loss).backward()
            total_loss += loss.item() * acc_step
            step_losses.append(loss.item() * acc_step)

            if (step + 1) % acc_step == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            # Cập nhật progress bar với loss hiện tại
            pbar.set_postfix({
                'loss': f"{loss.item() * acc_step:.4f}",
                'avg_loss': f"{total_loss / (step + 1):.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluation sau mỗi epoch
        model.eval()
        eval_total_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  Evaluating", ncols=120, leave=False):
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                labels = batch['labels'].to(model.device)
                with autocast('cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                eval_total_loss += outputs.loss.item()

        avg_eval_loss = eval_total_loss / len(val_loader)
        eval_losses.append(avg_eval_loss)

        # Generate predictions và tính P/R/F1 trên dev set
        predictions, gold_labels = generate_dev_predictions(
            model, tokenizer, val_data, batch_size=eval_batch_size
        )
        metrics = compute_coqe_metrics(predictions, gold_labels)

        # In kết quả epoch
        is_best = avg_eval_loss < best_eval_loss
        if is_best:
            best_eval_loss = avg_eval_loss

        print(f"\n{'='*60}")
        print(f"Epoch [{epoch+1}/{epochs}] Summary:")
        print(f"  Train Loss : {avg_train_loss:.5f}")
        print(f"  Eval  Loss : {avg_eval_loss:.5f}" + (" ← best" if is_best else ""))
        print(f"  LR         : {scheduler.get_last_lr()[0]:.2e}")
        print(f"{'='*60}")
        print_metrics_table(metrics, epoch=epoch+1)

    print("#" * 20 + " FINISH TRAINING " + "#" * 20)
    print(f"\nBest Eval Loss: {best_eval_loss:.5f}")
    print(f"Train Losses: {[f'{l:.5f}' for l in train_losses]}")
    print(f"Eval  Losses: {[f'{l:.5f}' for l in eval_losses]}")

def print_sample_data(dataset, tokenizer, num_samples=5):
    """In ra các mẫu input-output từ dataset để kiểm tra dữ liệu."""
    print("\nSample Input-Output Pairs:")
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        input_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        
        # Thay thế -100 bằng pad_token_id trước khi decode
        labels = sample['labels'].clone()
        labels[labels == -100] = tokenizer.pad_token_id
        label_text = tokenizer.decode(labels, skip_special_tokens=True)
        
        print(f"  Input: {input_text}")
        print(f"  Label: {label_text}")