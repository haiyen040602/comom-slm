from typing import Dict, List

_BASE_INSTRUCTION_EN = (
    "You are an information extraction model for comparative opinion mining. "
    "Given one sentence, extract all comparative opinion quintuples. "
    "Each quintuple has five slots: "
    "[S] subject, [O] object, [A] aspect, [P] predicate, [L] comparison label. "
    "Output ONLY tuples in this exact format: ([S] ... [O] ... [A] ... [P] ... [L] ...). "
    "If there are multiple tuples, separate them with ' ; '. "
    "Every extracted span for [S], [O], [A], and [P] must be copied verbatim from the original sentence. "
    "Do not paraphrase, normalize, translate, summarize, or invent any span. "
    "If a slot has no value, write [UNK] for it. "
    "If the sentence contains no comparison at all, output exactly: "
    "([S] [UNK] [O] [UNK] [A] [UNK] [P] [UNK] [L] [UNK]). "
    "Do not output any explanation or extra text."
)

_BASE_INSTRUCTION_VI = (
    "Bạn là một mô hình trích xuất thông tin cho bài toán khai thác quan điểm so sánh. "
    "Cho một câu, hãy trích xuất tất cả bộ năm thành phần so sánh trong câu (comparative quintuples). "
    "Mỗi quintuple có 5 thành phần: "
    "[S] là chủ thể (subject), [O] là đối tượng so sánh (object), [A] là một thuộc tính được so sánh (aspect), [P] là từ/cụm từ so sánh (comparative predicate), [L] là nhãn quan hệ so sánh (comparative label). "
    "Bạn chỉ được sinh kết quả theo đúng định dạng là ([S] ... [O] ... [A] ... [P] ... [L] ...). "
    "Nếu câu có nhiều có nhiều quan hệ so sánh, hãy sinh các quintuple được ngăn cách bằng ' ; '. "
    "Mọi cụm từ được trích xuất cho [S], [O], [A], [P] phải là nguyên văn xuất hiện trong câu gốc. "
    "Nếu đối tượng so sánh bị ẩn nhưng có thể suy luận rõ ràng từ ngữ cảnh trong câu, hãy sử dụng từ ngữ của đối tượng đó đã xuất hiện ở phần trước của câu. "
    "Không được diễn giải lại, chuẩn hóa lại, dịch, tóm tắt hay tự bịa thêm cụm từ. "
    "Nếu thành phần nào không có giá trị, điền [UNK]. "
    "Nếu câu không có quan hệ so sánh, trả về kết quả đúng như sau: "
    "([S] [UNK] [O] [UNK] [A] [UNK] [P] [UNK] [L] [UNK]). "
    "Không được thêm giải thích hay văn bản nào khác."
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

_T5_LABEL_NOTE_VI = (
    "Giá trị hợp lệ của [L]: Better, Worse, Equal, Different. "
    "Chỉ dùng [UNK] khi câu không có quan hệ so sánh."
)

_VCOM_LABEL_NOTE_VI = (
    "Giá trị hợp lệ của [L]: COM+, COM-, COM, SUP+, SUP-, SUP, EQL, DIF. "
    "COM+ là so sánh tích cực, chủ thể tốt hơn đối tượng. "
    "COM- là so sánh tiêu cực, chủ thể kém hơn đối tượng. "
    "COM là có quan hệ so sánh hơn nhưng không rõ hướng cảm xúc. "
    "SUP+ là so sánh bậc nhất theo hướng tích cực. "
    "SUP- là so sánh bậc nhất theo hướng tiêu cực. "
    "SUP là so sánh bậc nhất nhưng không rõ cực tính. "
    "EQL là so sánh tương đương, không khác biệt đáng kể. "
    "DIF là khác biệt nhưng không thể hiện rõ tốt hơn hay kém hơn. "
    "Chỉ dùng [UNK] khi câu không có quan hệ so sánh."
)

_USER_CONTRACT_EN = (
    "Task: Extract all comparative opinion quintuple(s) from the input sentence.\n\n"
    "Output contract:\n"
    "- Output only tuple(s) in this exact format: ([S] ... [O] ... [A] ... [P] ... [L] ...).\n"
    "- If there are multiple tuples, separate them with ' ; '.\n"
    "- Every span in [S], [O], [A], and [P] must appear verbatim in the original sentence.\n"
    "- Do not paraphrase, rewrite, translate, normalize, or invent spans.\n"
    "- If a slot is missing, use [UNK].\n"
    "- If the sentence has no comparative meaning, output exactly: ([S] [UNK] [O] [UNK] [A] [UNK] [P] [UNK] [L] [UNK]).\n"
    "- Do not output explanations, reasoning, bullets, or markdown."
)

_USER_CONTRACT_VI = (
    "Nhiệm vụ: Trích xuất tất cả quintuple quan điểm so sánh từ câu đầu vào và đảm bảo các ràng buộc sau:\n"
    "- Chỉ xuất tuple theo đúng định dạng: ([S] ... [O] ... [A] ... [P] ... [L] ...).\n"
    "- Nếu câu có nhiều comparative tuple, ngăn cách bằng ' ; '.\n"
    "- Mọi cụm từ ở [S], [O], [A], [P] phải xuất hiện nguyên văn trong câu gốc.\n"
    "- Nếu đối tượng so sánh bị ẩn nhưng có thể suy luận rõ ràng từ ngữ cảnh trong câu, hãy sử dụng từ ngữ của đối tượng đó đã xuất hiện ở phần trước của câu.\n"
    "- Không được diễn giải lại, viết lại, dịch, chuẩn hóa hay tự thêm cụm từ.\n"
    "- Nếu một thành phần bị thiếu, dùng [UNK].\n"
    "- Nếu câu không có quan hệ so sánh, chỉ được trả về kết quả: ([S] [UNK] [O] [UNK] [A] [UNK] [P] [UNK] [L] [UNK]).\n"
    "- Không được in giải thích, lập luận, hay văn bản khác trong câu trả lời."
)


def _is_vi(dataset: str, language: str) -> bool:
    return dataset == "vcom-data" or language == "vi"


def _base_instruction(is_vi: bool) -> str:
    return _BASE_INSTRUCTION_VI if is_vi else _BASE_INSTRUCTION_EN


def _label_note(is_vi: bool, dataset: str) -> str:
    if is_vi:
        return _VCOM_LABEL_NOTE_VI if dataset == "vcom-data" else _T5_LABEL_NOTE_VI
    return _VCOM_LABEL_NOTE if dataset == "vcom-data" else _T5_LABEL_NOTE


def _strategy_note(is_vi: bool, strategy: str) -> str:
    if strategy != "cot":
        return ""
    if is_vi:
        return (
            "Hãy suy luận nội bộ theo từng bước nhưng không được in ra chain-of-thought. "
            "Chỉ xuất tuple cuối cùng đúng định dạng đã quy định ở trên."
        )
    return (
        "Reason internally step by step, but never reveal chain-of-thought. "
        "Only output final tuples in the exact required format above."
    )


def _user_contract(is_vi: bool) -> str:
    return _USER_CONTRACT_VI if is_vi else _USER_CONTRACT_EN


def _zero_shot_user_en(sentence: str) -> str:
    return (
        f"{_user_contract(False)}\n\n"
        "Follow the instruction and infer the tuple(s) directly from the sentence.\n\n"
        f"Input sentence: {sentence}\n"
        "Output:"
    )


def _zero_shot_user_vi(sentence: str) -> str:
    return (
        f"{_user_contract(True)}\n\n"
        "Hãy bám đúng hướng dẫn và suy ra tuple trực tiếp từ câu: "
        f"{sentence}\n"
        "Kết quả:"
    )


def _few_shot_user_en(sentence: str) -> str:
    # 3 examples: one tuple, multiple tuples, and no tuple.
    return (
        f"{_user_contract(False)}\n\n"
        "Learn the output style from the examples, then solve the final sentence.\n\n"
        "Example 1 (one quintuple)\n"
        "Sentence: The file-size gets even bigger if you shoot in RAW format instead of JPEG format.\n"
        "Output: ([S] RAW format [O] JPEG format [A] file-size [P] bigger [L] Better)\n\n"
        "Example 2 (multiple quintuples)\n"
        "Sentence: The picture quality, speed, processing, even the sound is better when you shoot an image.\n"
        "Output: ([S] [UNK] [O] [UNK] [A] picture quality [P] better [L] Better) ; ([S] [UNK] [O] [UNK] [A] speed [P] better [L] Better) ; ([S] [UNK] [O] [UNK] [A] processing [P] better [L] Better) ; ([S] [UNK] [O] [UNK] [A] sound [P] better [L] Better)\n\n"
        "Example 3 (no quintuple)\n"
        "Sentence: The pictures are truly professional quality.\n"
        "Output: ([S] [UNK] [O] [UNK] [A] [UNK] [P] [UNK] [L] [UNK])\n\n"
        f"Sentence: {sentence}\n"
        "Output:"
    )


def _few_shot_user_vi(sentence: str) -> str:
    # 3 examples: one tuple, multiple tuples, and no tuple.
    return (
        f"{_user_contract(True)}\n\n"
        "Hãy học cách xuất kết quả từ các ví dụ rồi xử lý câu cuối cùng.\n\n"
        "Ví dụ 1 (1 quintuple)\n"
        "Câu: Tương tự, thì ống kính góc rộng không có quá nhiều sự khác biệt so với ống kính chính.\n"
        "Kết quả: ([S] ống kính góc rộng [O] ống kính chính [A] [UNK] [P] không có quá nhiều sự khác biệt [L] EQL)\n\n"
        "Ví dụ 2 (nhiều quintuple)\n"
        "Câu: Bên cạnh đó, iPhone 14 được nâng cấp bộ nhớ lên đến 6GB RAM cao hơn iPhone 13 đến 2GB RAM, cho khả năng đa nhiệm tốt hơn.\n"
        "Kết quả: ([S] iPhone 14 [O] iPhone 13 [A] bộ nhớ [P] cao hơn [L] COM+) ; ([S] iPhone 14 [O] iPhone 13 [A] khả năng đa nhiệm [P] tốt hơn [L] COM+)\n\n"
        "Ví dụ 3 (không có quintuple)\n"
        "Câu: Bạn có thể selfie và sử dụng ở bể bơi mà không hề sợ bị hỏng máy.\n"
        "Kết quả: ([S] [UNK] [O] [UNK] [A] [UNK] [P] [UNK] [L] [UNK])\n\n"
        f"Câu: {sentence}\n"
        "Kết quả:"
    )


def _cot_user_en(sentence: str) -> str:
    return (
        f"{_user_contract(False)}\n\n"
        "Reason internally using this order: identify comparative subject, identify comparative object, identify aspect, identify comparative predicate and polarity, then form final tuple(s).\n"
        "Do not reveal the reasoning. Only output the final tuple(s).\n\n"
        f"Input sentence: {sentence}\n"
        "Output:"
    )


def _cot_user_vi(sentence: str) -> str:
    return (
        f"{_user_contract(True)}\n\n"
        "Hãy suy luận nội bộ theo thứ tự: xác định chủ thể so sánh, xác định đối tượng so sánh, xác định thuộc tính, xác định từ/cụm từ so sánh và cực tính, rồi tạo tuple cuối cùng.\n"
        "Không được in ra lập luận. Chỉ xuất tuple cuối cùng.\n\n"
        f"Câu văn cần xử lý: {sentence}\n"
        "Kết quả:"
    )


def build_messages(
    sentence: str,
    language: str = "auto",
    dataset: str = "",
    strategy: str = "zero-shot",
) -> List[Dict[str, str]]:
    """Build chat messages for a single sentence.

    Parameters
    ----------
    sentence : the raw input sentence.
    language : 'en', 'vi', or 'auto'.
    dataset  : dataset name used to select the correct label set
               ('t5-camera-coqe-data' → 4-class, 'vcom-data' → 8-class).
    """
    is_vi = _is_vi(dataset, language)

    if strategy not in {"zero-shot", "few-shot", "cot"}:
        raise ValueError("strategy must be one of: zero-shot, few-shot, cot")

    system_content = "\n".join(
        p for p in [_base_instruction(is_vi), _label_note(is_vi, dataset), _strategy_note(is_vi, strategy)] if p
    )

    if strategy == "zero-shot":
        user_content = _zero_shot_user_vi(sentence) if is_vi else _zero_shot_user_en(sentence)
    elif strategy == "few-shot":
        user_content = _few_shot_user_vi(sentence) if is_vi else _few_shot_user_en(sentence)
    else:
        user_content = _cot_user_vi(sentence) if is_vi else _cot_user_en(sentence)

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]
