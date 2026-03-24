import re

import torch
from torch.utils.data import Dataset

OUTPUT_END_MARKER = "<|tuple_end|>"
PROMPT_STYLE_DIRECT = "direct"
PROMPT_STYLE_COT = "cot"
ELEM_NAMES = ['S', 'O', 'A', 'P', 'L']
DEFAULT_EMPTY_TUPLE = "([S] [UNK] [O] [UNK] [A] [UNK] [P] [UNK] [L] [UNK])"
DIRECT_TASK_INSTRUCTION = (
    "Extract comparative opinion tuples from the sentence. "
    "Use [S] for subject, [O] for object, [A] for aspect, [P] for predicate, and [L] for label. "
    "Return only tuples in this exact format: ([S] ... [O] ... [A] ... [P] ... [L] ...). "
    "Allowed labels for [L]: Better, Worse, Equal, Different, [UNK]. "
    f"If there is no comparison, output exactly {DEFAULT_EMPTY_TUPLE}. "
    "If there are multiple tuples, separate them with ' ; '. Do not explain. Do not add extra words."
)
COT_TASK_INSTRUCTION = (
    "Extract comparative opinion tuples from the sentence. "
    "Use [S] for subject, [O] for object, [A] for aspect, [P] for predicate, and [L] for label. "
    "Write exactly two sections. First write 'Analysis:' with one short reasoning sentence. "
    "Then write 'Result:' followed only by tuples in this exact format: ([S] ... [O] ... [A] ... [P] ... [L] ...). "
    "Allowed labels for [L]: Better, Worse, Equal, Different, [UNK]. "
    f"If there is no comparison, write 'Result: {DEFAULT_EMPTY_TUPLE}'. "
    "If there are multiple tuples, separate them with ' ; '. Do not add extra sections or commentary."
)
TUPLE_PATTERN = re.compile(
    r'\[S\]\s*(.*?)\s*\[O\]\s*(.*?)\s*\[A\]\s*(.*?)\s*\[P\]\s*(.*?)\s*\[L\]\s*(.*?)(?=\)|\n|;|$)',
    re.DOTALL,
)


def get_task_instruction(prompt_style=PROMPT_STYLE_DIRECT):
    if prompt_style == PROMPT_STYLE_COT:
        return COT_TASK_INSTRUCTION
    return DIRECT_TASK_INSTRUCTION


def build_prompt(input_text, prompt_style=PROMPT_STYLE_DIRECT):
    instruction = get_task_instruction(prompt_style)
    return f"Instruction: {instruction}\nInput: {input_text}\nOutput:"


def _parse_target_tuple(text):
    text = text.strip().strip('()')
    match = TUPLE_PATTERN.search(text)
    if match:
        return tuple(item.strip() for item in match.groups())
    return None


def _parse_target_output(text):
    tuples = []
    for part in text.split(';'):
        parsed = _parse_target_tuple(part.strip())
        if parsed:
            tuples.append(parsed)
    return tuples


def _is_all_unk_tuple(tuple_value):
    return len(tuple_value) >= 5 and all((slot or '').strip() == '[UNK]' for slot in tuple_value[:5])


def _build_cot_analysis(target_text):
    tuples = _parse_target_output(target_text)
    if not tuples:
        return "No comparison is expressed."

    analyses = []
    for tuple_value in tuples:
        if _is_all_unk_tuple(tuple_value):
            analyses.append("No comparison is expressed.")
            continue

        subject, obj, aspect, predicate, label = tuple_value
        analyses.append(
            f"Compare {subject} vs {obj} on {aspect}; predicate: {predicate}; label: {label}."
        )
    return " | ".join(analyses)


def build_training_target(target_text, prompt_style=PROMPT_STYLE_DIRECT):
    target_text = target_text.strip() if target_text else ""
    if prompt_style == PROMPT_STYLE_COT:
        analysis = _build_cot_analysis(target_text or DEFAULT_EMPTY_TUPLE)
        result_text = target_text or DEFAULT_EMPTY_TUPLE
        return f"Analysis: {analysis}\nResult: {result_text}"
    return target_text

class CausalLMDataset(Dataset):
    """Dataset for Causal Language Models (Phi, Qwen)
    
    For causal LM, we combine input and output into a single sequence:
    Format: "Input: [sentence] Output: [label]"
    """
    def __init__(self, tokenizer, inputs=None, targets=None, max_len=256, prompt_style=PROMPT_STYLE_DIRECT):
        self.tokenizer = tokenizer
        self.inputs = inputs or []
        self.targets = targets or []
        self.max_len = max_len
        self.prompt_style = prompt_style
        self.formatted_targets = []
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
        
        self.formatted_targets = []

        for i in range(len(inputs)):
            input_text = ' '.join(inputs[i]) if isinstance(inputs[i], list) else inputs[i]
            raw_target_text = ' '.join(targets[i]) if isinstance(targets[i], list) else targets[i]
            formatted_target = build_training_target(raw_target_text, prompt_style=self.prompt_style)
            self.formatted_targets.append(formatted_target)

            target_text = formatted_target
            if target_text:
                # Add an explicit end marker so the model learns where to stop.
                target_text = f"{target_text} {OUTPUT_END_MARKER}"
            
            # Handle both training (non-empty) and test (empty) data
            combined_text = f"{build_prompt(input_text, prompt_style=self.prompt_style)} {target_text}"
            
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
            input_part = build_prompt(input_text, prompt_style=self.prompt_style)
            input_encoded = self.tokenizer(input_part, return_tensors="pt")
            input_ids_len = input_encoded['input_ids'].shape[-1]
            
            labels = input_ids.clone()
            # Only compute loss on output tokens
            labels[:input_ids_len] = -100
            labels[attention_mask == 0] = -100
            
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