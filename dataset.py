import torch
from torch.utils.data import Dataset

OUTPUT_END_MARKER = "<|tuple_end|>"

class CausalLMDataset(Dataset):
    """Dataset for Causal Language Models (Phi, Qwen)
    
    For causal LM, we combine input and output into a single sequence:
    Format: "Input: [sentence] Output: [label]"
    """
    def __init__(self, tokenizer, inputs=None, targets=None, max_len=256):
        self.tokenizer = tokenizer
        self.inputs = inputs or []
        self.targets = targets or []
        self.max_len = max_len
        self.encoded_data = self.encode(self.inputs, self.targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.encoded_data[idx]

    def encode(self, inputs=[], targets=[]):
        """Encode input-output pairs for causal LM
        
        For training: "Input: sentence\nOutput: structured_output"
        For test: "Input: sentence\nOutput: " (empty for generation)
        """
        encoded_data = []
        
        for i in range(len(inputs)):
            input_text = ' '.join(inputs[i]) if isinstance(inputs[i], list) else inputs[i]
            target_text = ' '.join(targets[i]) if isinstance(targets[i], list) else targets[i]
            if target_text:
                # Add an explicit end marker so the model learns where to stop.
                target_text = f"{target_text} {OUTPUT_END_MARKER}"
            
            # Handle both training (non-empty) and test (empty) data
            combined_text = f"Input: {input_text}\nOutput: {target_text}"
            
            # Tokenize combined text
            encoded = self.tokenizer(
                combined_text,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoded['input_ids'].squeeze()
            attention_mask = encoded['attention_mask'].squeeze()
            
            # Create labels: -100 for input part, token_ids for output part
            # For test data with empty output, mark input as -100
            input_part = f"Input: {input_text}\nOutput:"
            input_encoded = self.tokenizer(input_part, return_tensors="pt")
            input_ids_len = input_encoded['input_ids'].shape[-1]
            
            labels = input_ids.clone()
            # Only compute loss on output tokens
            labels[:input_ids_len] = -100
            
            # Handle scalar tensor shape
            if input_ids.dim() == 0:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask.dim() == 0:
                attention_mask = attention_mask.unsqueeze(0)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            
            encoded_data.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            })
        
        return encoded_data