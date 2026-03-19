import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def infer(dataset, model, tokenizer, batch_size, max_seq_length=256, name="eval", verbose=False):
    """Inference for causal LM - generates predictions"""
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    
    inputs, outputs, targets = [], [], []
    average_loss = 0
    
    model.eval()
    
    with torch.no_grad():
        if name == "eval":
            # Evaluation: compute loss
            total_loss = 0
            num_batches = len(data_loader)
            
            for batch in tqdm(data_loader, desc="Evaluating", disable=not verbose):
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                labels = batch['labels'].to(model.device)
                
                outputs_batch = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs_batch.loss
                total_loss += loss.item()
            
            average_loss = total_loss / num_batches
        else:
            # Inference: generate outputs
            for batch in tqdm(data_loader, desc=f"Inferencing ({name})", disable=not verbose):
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                
                # Generate
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_seq_length,
                    num_beams=1,
                    early_stopping=True,
                    do_sample=False
                )
                
                # Decode
                for i, gen_ids in enumerate(generated_ids):
                    text = tokenizer.decode(gen_ids, skip_special_tokens=False)
                    input_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    
                    # Extract output part after "Output:"
                    if "Output:" in text:
                        output_text = text.split("Output:")[-1].strip()
                    else:
                        output_text = text
                    
                    inputs.append(input_text)
                    outputs.append(output_text)
    
    return average_loss, inputs, outputs, targets