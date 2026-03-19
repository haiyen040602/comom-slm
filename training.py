import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler

def train(model, tokenizer, train_data, val_data, epochs, lr, train_batch_size, eval_batch_size, acc_step=4):
    """Training function for Causal LM with memory optimization"""
    print("#" * 20 + " BEGIN TRAINING " + "#" * 20)
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    print("✓ Gradient checkpointing enabled")
    
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

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / acc_step

            scaler.scale(loss).backward()
            total_loss += loss.item() * acc_step

            if (step + 1) % acc_step == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader)}")

    print("#" * 20 + " FINISH TRAINING " + "#" * 20)