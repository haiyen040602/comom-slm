from typing import Dict, List

_BASE_INSTRUCTION = (
    "You are an information extraction model for comparative opinion mining. "
    "Given one sentence, extract all comparative opinion quintuples. "
    "Each quintuple has five slots: "
    "[S] subject, [O] object, [A] aspect, [P] predicate, [L] comparison label. "
    "Output ONLY tuples in this exact format: ([S] ... [O] ... [A] ... [P] ... [L] ...). "
    "If there are multiple tuples, separate them with ' ; '. "
    "If a slot has no value, write [UNK] for it. "
    "If the sentence contains no comparison at all, output exactly: "
    "([S] [UNK] [O] [UNK] [A] [UNK] [P] [UNK] [L] [UNK]). "
    "Do not output any explanation or extra text."
)

# English dataset: 4-class labels
_T5_LABEL_NOTE = (
    "Allowed values for [L]: Better, Worse, Equal, Different. "
    "Use [UNK] only when no comparison is present."
)

# Vietnamese dataset: 8-class labels
_VCOM_LABEL_NOTE = (
    "Allowed values for [L]: COM+, COM-, COM, SUP+, SUP-, SUP, EQL, DIF. "
    "COM+ = positive comparison (subject is better than object). "
    "COM- = negative comparison (subject is worse than object). "
    "COM  = comparison (direction unspecified). "
    "SUP+ = positive superlative. "
    "SUP- = negative superlative. "
    "SUP  = superlative (polarity unspecified). "
    "EQL  = equal / no significant difference. "
    "DIF  = different (no clear better/worse). "
    "Use [UNK] only when no comparison is present."
)


def build_messages(
    sentence: str,
    language: str = "auto",
    dataset: str = "",
) -> List[Dict[str, str]]:
    """Build chat messages for a single sentence.

    Parameters
    ----------
    sentence : the raw input sentence.
    language : 'en', 'vi', or 'auto'.
    dataset  : dataset name used to select the correct label set
               ('t5-camera-coqe-data' → 4-class, 'vcom-data' → 8-class).
    """
    # Choose label note based on dataset (fallback: language heuristic)
    if dataset == "vcom-data" or language == "vi":
        label_note = _VCOM_LABEL_NOTE
        lang_note  = "The sentence is in Vietnamese."
    else:
        label_note = _T5_LABEL_NOTE
        lang_note  = "The sentence is in English." if language == "en" else ""

    system_content = f"{_BASE_INSTRUCTION}\n{label_note}"
    user_content   = f"{lang_note}\nSentence: {sentence}\nOutput:".strip()

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]
