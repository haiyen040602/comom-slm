from typing import Dict, List

_BASE_INSTRUCTION_EN = (
    "You are an information extraction model for comparative opinion mining. "
    # "Your task is to identify and extract comparative opinions from sentences. "
    # "Given one sentence, if it contains no comparison, "
    # "output exactly: ([S] [UNK] [O] [UNK] [A] [UNK] [P] [UNK] [L] [UNK]). "
    # "If it contains a comparison, extract all comparative opinion quintuples. "
    # "Each quintuple has five slots: "
    # "[S] is comparative subject, [O] is comparative object, [A] is comparative aspect, [P] is comparative predicate, [L] is comparative label. "
    # "Output ONLY tuples in this exact format: ([S] ... [O] ... [A] ... [P] ... [L] ...). "
    # "If there are multiple comparisons, separate them with ' ; '. "
    # "Every extracted span for [S], [O], [A], and [P] must be copied verbatim from the original sentence. "
    # "Do not paraphrase, normalize, translate, summarize, or invent any span. "
    # "If a slot has no value, write [UNK] for it. "
    # "Do not output any explanation or extra text."
)

_BASE_INSTRUCTION_VI = (
    "Bạn là một mô hình trích xuất thông tin cho bài toán khai thác quan điểm so sánh. "
    # "Nhiệm vụ của bạn là phát hiện và trích xuất các quan điểm so sánh từ câu văn.\n"
    # "Cho một câu văn, nếu trong câu không có quan hệ so sánh, hãy trả về kết quả đúng như sau: "
    # "([S] [UNK] [O] [UNK] [A] [UNK] [P] [UNK] [L] [UNK]). "
    # "Nếu trong câu có quan hệ so sánh, hãy trích xuất tất cả bộ năm thành phần so sánh trong câu (comparative quintuples). "
    # "Mỗi quintuple có 5 thành phần: "
    # "[S] là chủ thể (subject), [O] là đối tượng so sánh (object), [A] là một thuộc tính được so sánh (aspect), [P] là từ/cụm từ so sánh (comparative predicate), [L] là nhãn quan hệ so sánh (comparative label). "
    # "Bạn chỉ được sinh kết quả theo đúng định dạng là ([S] ... [O] ... [A] ... [P] ... [L] ...). "
    # "Nếu câu có nhiều có nhiều quan hệ so sánh, hãy sinh các quintuple được ngăn cách bằng ' ; '. "
    # "Mọi cụm từ được trích xuất cho [S], [O], [A], [P] phải là nguyên văn xuất hiện trong câu gốc. "
    # "Nếu đối tượng so sánh bị ẩn nhưng có thể suy luận rõ ràng từ ngữ cảnh trong câu, hãy sử dụng từ ngữ của đối tượng đó đã xuất hiện ở phần trước của câu. "
    # "Không được diễn giải lại, chuẩn hóa lại, dịch, tóm tắt hay tự bịa thêm cụm từ. "
    # "Nếu thành phần nào không có giá trị, điền [UNK]. "
    # "Không được thêm giải thích hay văn bản nào khác."
)

_USER_CONTRACT_EN = (
    "Your task is to extract all comparative opinion quintuple(s) from the input sentence if it contains any comparative meaning.\n"
    "You must follow these rules:\n"
    "- Output only tuple(s) in this exact format: ([S] ... [O] ... [A] ... [P] ... [L] ...).\n"
    "- [S] is the comparative subject, [O] is the comparative object, [A] is a comparative aspect, [P] is the comparative predicate, and [L] is the comparative label.\n"
    "- If the sentence has no comparative meaning and is just a simple description with no comparative or superlative intent, output exactly: ([S] [UNK] [O] [UNK] [A] [UNK] [P] [UNK] [L] [UNK]).\n"
    "- If there are multiple tuples, separate them with ' ; '.\n"
    "- Every span in [S], [O], [A], and [P] must appear verbatim in the original sentence.\n"
    "- Allowed values for [L]: Better, Worse, Equal, Different. " \
    "Better = the comparative subject is better than the comparative object. " \
    "Worse = the comparative subject is worse than the comparative object. " \
    "Equal = the comparative subject and comparative object are equal or have no significant difference. " \
    "Different = the comparative subject and comparative object are different but it's unclear which one is better.\n" \
    "- If a comparative component is implicit but can be clearly inferred from the context, use the specific words for that entity that appeared earlier in the sentence.\n"
    "- If a comparative component is missing and cannot be clearly inferred, use [UNK].\n"
    "- Do not paraphrase, rewrite, translate, normalize, or invent spans.\n"
    "- Do not output explanations, reasoning, bullets, or markdown."
)

_USER_CONTRACT_VI = (
    "Cho một câu văn, nhiệm vụ của bạn là phân tích và trích xuất thông tin các phép so sánh (quintuple) trong câu nếu câu có chứa ý so sánh. Bạn cần đảm bảo các ràng buộc sau:\n"
    "- Chỉ sinh ra kết quả theo đúng định dạng tuple: ([S] ... [O] ... [A] ... [P] ... [L] ...).\n"
    "- Trong đó [S] là chủ thể (subject), [O] là đối tượng so sánh (object), [A] là một thuộc tính được so sánh (aspect), [P] là từ/cụm từ so sánh (comparative predicate), [L] là nhãn quan hệ so sánh (comparative label). \n"
    "- Nếu trong câu không có quan hệ so sánh, chỉ cần trả về kết quả đúng như sau: ([S] [UNK] [O] [UNK] [A] [UNK] [P] [UNK] [L] [UNK]).\n"
    "- Nếu câu có nhiều quan hệ so sánh, ngăn cách các tuple bằng ' ; '.\n"
    "- Mọi cụm từ được trích xuất cho [S], [O], [A], [P] phải xuất hiện nguyên văn trong câu gốc.\n"
    "- Nếu có quan hệ so sánh, giá trị hợp lệ của [L] là: COM+, COM-, COM, SUP+, SUP-, SUP, EQL, DIF. "
    "COM+ là so sánh hơn theo hướng tích cực, [S] được đánh giá tốt hơn [O]. "
    "COM- là so sánh hơn theo hướng tiêu cực, [S] được đánh giá kém hơn [O]. "
    "COM là so sánh hơn nhưng không rõ hướng cảm xúc. "
    "SUP+ là so sánh bậc nhất theo hướng tích cực, [S] được đánh giá tốt nhất. "
    "SUP- là so sánh bậc nhất theo hướng tiêu cực, [S] được đánh giá kém nhất. "
    "SUP là so sánh bậc nhất nhưng không rõ hướng cảm xúc. "
    "EQL là so sánh tương đương, không khác biệt đáng kể, [S] và [O] được đánh giá tương đương. "
    "DIF là khác biệt, [S] và [O] được đánh giá khác biệt nhưng không thể hiện rõ tốt hơn hay kém hơn. \n"
    "- Nếu thành phần so sánh bị ẩn nhưng có thể suy luận rõ ràng từ ngữ cảnh trong câu, hãy sử dụng từ ngữ của đối tượng đó đã xuất hiện ở phần trước của câu.\n"
    "- Nếu một thành phần bị thiếu, không tìm được trong câu văn gốc và không thể suy luận rõ ràng, dùng [UNK], không tự thêm giá trị.\n"
    "- Không được diễn giải lại, viết lại, dịch, chuẩn hóa hay tự thêm cụm từ.\n"
    "- Không được thêm giải thích, lập luận, hay văn bản khác trong câu trả lời."
    # "Nhiệm vụ của bạn là phát hiện và trích xuất các quan điểm so sánh từ câu văn.\n"
    # "Cho một câu văn, nếu trong câu không có quan hệ so sánh, hãy trả về kết quả đúng như sau: "
    # "([S] [UNK] [O] [UNK] [A] [UNK] [P] [UNK] [L] [UNK]). "
    # "Nếu trong câu có quan hệ so sánh, hãy trích xuất tất cả bộ năm thành phần so sánh trong câu (comparative quintuples). "
    # "Mỗi quintuple có 5 thành phần: "
    # "[S] là chủ thể (subject), [O] là đối tượng so sánh (object), [A] là một thuộc tính được so sánh (aspect), [P] là từ/cụm từ so sánh (comparative predicate), [L] là nhãn quan hệ so sánh (comparative label). "
    # "Bạn chỉ được sinh kết quả theo đúng định dạng là ([S] ... [O] ... [A] ... [P] ... [L] ...). "
    # "Nếu câu có nhiều có nhiều quan hệ so sánh, hãy sinh các quintuple được ngăn cách bằng ' ; '. "
    # "Mọi cụm từ được trích xuất cho [S], [O], [A], [P] phải là nguyên văn xuất hiện trong câu gốc. "
    # "Nếu đối tượng so sánh bị ẩn nhưng có thể suy luận rõ ràng từ ngữ cảnh trong câu, hãy sử dụng từ ngữ của đối tượng đó đã xuất hiện ở phần trước của câu. "
    # "Không được diễn giải lại, chuẩn hóa lại, dịch, tóm tắt hay tự bịa thêm cụm từ. "
    # "Nếu thành phần nào không có giá trị, điền [UNK]. "
    # "Không được thêm giải thích hay văn bản nào khác."
)

_SHORT_USER_EN = "Extract comparative quintuples for this sentence:\n{sentence}"
_SHORT_USER_VI = "Trích xuất các thành phần so sánh (quintuple) trong câu sau:\n{sentence}"

_SHORT_USER_JSON_EN = "Sentence: {sentence}"
_SHORT_USER_JSON_VI = "Câu: {sentence}"

_JSON_SCHEMA_NOTE_EN = (
    "Return JSON only (no markdown, no explanation) with schema: "
    '{"comparisons": [{"label": "...", "subject": null, "object": null, "aspect": null, "predicate": null}]}. '
    "If there is no comparison, return exactly: {\"comparisons\": []}."
)

_JSON_SCHEMA_NOTE_VI = (
    "Chỉ trả về JSON (không markdown, không giải thích) theo schema: "
    '{"comparisons": [{"label": "...", "subject": null, "object": null, "aspect": null, "predicate": null}]}. '
    "Nếu không có quan hệ so sánh, trả về đúng: {\"comparisons\": []}."
)

_JSON_SYSTEM_SCAFFOLD_VI = (
    "Bạn là một mô hình trích xuất thông tin cho bài toán khai thác quan điểm so sánh. \n"
    "Cho một câu văn, nhiệm vụ của bạn là phân tích và trích xuất thông tin các phép so sánh (quintuple) trong câu nếu câu có chứa ý so sánh. Bạn cần đảm bảo các ràng buộc sau:\n"
    "- Phân tích, trích xuất 5 thành phần của quan hệ so sánh: chủ thể (subject), đối tượng (object), thuộc tính (aspect), từ/cụm từ so sánh (predicate), nhãn quan hệ (label).\n"
    "- Một câu có thể có một hoặc nhiều quan hệ so sánh; phải trích xuất đầy đủ.\n"
    "- subject, object, aspect, predicate phải là cụm từ nguyên văn xuất hiện trong câu gốc. \n"
    "- Nếu một thành phần bị ẩn nhưng có thể suy luận rõ ràng từ ngữ cảnh trong câu, hãy sử dụng từ ngữ của thành phần đó đã xuất hiện ở phần trước của câu. \n"
    "- Nếu thành phần nào không có giá trị và không thể suy luận rõ ràng, điền null. \n"
    # "Quy tắc lược bỏ / đồng tham chiếu (ellipsis/coreference):\n"
    # "- Nếu một mệnh đề hàm ý so sánh nhưng bị thiếu một thực thể (vd: 'rẻ hơn hẳn', 'nhanh hơn nhiều', 'cũng ổn hơn'):\n"
    # "  1) Gán đại từ về thực thể được nếu rõ gần nhất.\n"
    # "  2) Tái sử dụng đối thủ (competitor) được nếu rõ gần nhất trước đó trong câu.\n"
    # "  3) Nếu không thể suy luận chắc chắn thì để subject/object = null, không tự bịa thêm thực thể.\n\n"
    "Các nhãn label hợp lệ:\n"
    "{label_rules}\n\n"
    "Output JSON schema:\n"
    "{\n"
    "  \"comparisons\": [\n"
    "    {\n"
    "      \"label\": \"{label_union}\",\n"
    "      \"subject\": string hoặc null,\n"
    "      \"object\": string hoặc null,\n"
    "      \"aspect\": string hoặc null,\n"
    "      \"predicate\": string hoặc null\n"
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Nếu câu có nhiều quan hệ so sánh, hãy trích xuất tất cả và trả về trong mảng comparisons.\n"
    "Nếu câu không có quan hệ so sánh nào, trả về đúng: {\"comparisons\": []}.\n"
    "Chỉ trả về JSON hợp lệ, không markdown, không giải thích."
)

_JSON_SYSTEM_SCAFFOLD_EN = (
    "You are an information extraction model for comparative opinion mining. \n"
    "Given one sentence, your task is to analyze and extract all comparative opinion quintuples if the sentence contains any comparative meaning. You must follow these rules:\n"
    "- Analyze and extract 5 components of each comparison: comparative subject, comparative object, comparative aspect, comparative predicate, and comparative label.\n"
    "- A sentence may contain one or multiple relations; extract all of them.\n"
    "- subject, object, aspect, predicate must be verbatim spans from the original sentence. \n"
    "- If a component is implicit but can be clearly inferred from the context in the sentence, use the specific words for that entity that appeared earlier in the sentence. \n"
    "- If a component has no value and cannot be clearly inferred, use null. \n"
    "Allowed labels:\n"
    "{label_rules}\n\n"
    "Output JSON schema:\n"
    "{\n"
    "  \"comparisons\": [\n"
    "    {\n"
    "      \"label\": \"{label_union}\",\n"
    "      \"subject\": string or null,\n"
    "      \"object\": string or null,\n"
    "      \"aspect\": string or null,\n"
    "      \"predicate\": string or null\n"
    "    }\n"
    "  ]\n"
    "}\n\n"
    "If there are multiple comparisons, extract all and return in the comparisons array.\n"
    "If no comparison exists, return exactly: {\"comparisons\": []}.\n"
    "Return valid JSON only (no markdown, no explanation)."
)


def _json_labels(dataset: str) -> List[str]:
    if dataset == "vcom-data":
        return ["COM+", "COM-", "COM", "SUP+", "SUP-", "SUP", "EQL", "DIF"]
    return ["Better", "Worse", "Equal", "Different"]


def _json_label_rules(dataset: str, is_vi: bool) -> str:
    if dataset == "vcom-data":
        if is_vi:
            return "\n".join(
                [
                    "- COM+: so sánh hơn theo hướng tích cực, subject được đánh giá tốt hơn object.",
                    "- COM-: so sánh theo hướng tiêu cực, subject được đánh giá kém hơn object.",
                    "- COM: có quan hệ so sánh hơn nhưng không rõ hướng tốt/xấu.",
                    "- SUP+: so sánh bậc nhất theo hướng tích cực, subject được đánh giá tốt nhất.",
                    "- SUP-: so sánh bậc nhất theo hướng tiêu cực, subject được đánh giá kém nhất.",
                    "- SUP: so sánh bậc nhất nhưng không rõ cực tính.",
                    "- EQL: tương đương, ngang bằng, không khác biệt đáng kể.",
                    "- DIF: khác biệt nhưng không thể hiện rõ hơn/kém.",
                ]
            )
        return "\n".join(
            [
                "- COM+: positive comparison; subject is better than object.",
                "- COM-: negative comparison; subject is worse than object.",
                "- COM: comparison exists but the direction is unclear.",
                "- SUP+: positive superlative.",
                "- SUP-: negative superlative.",
                "- SUP: superlative with unclear polarity.",
                "- EQL: equal / no significant difference.",
                "- DIF: different without a clear better/worse direction.",
            ]
        )

    if is_vi:
        return "\n".join(
            [
                "- Better: subject tốt hơn object.",
                "- Worse: subject kém hơn object.",
                "- Equal: subject và object tương đương, không khác biệt đáng kể.",
                "- Different: subject và object khác nhau nhưng không rõ bên nào tốt hơn.",
            ]
        )
    return "\n".join(
        [
            "- Better: subject is better than object.",
            "- Worse: subject is worse than object.",
            "- Equal: subject and object are equal or have no significant difference.",
            "- Different: subject and object are different but neither side is clearly better.",
        ]
    )


def _json_system_scaffold(dataset: str, is_vi: bool) -> str:
    labels = _json_labels(dataset)
    template = _JSON_SYSTEM_SCAFFOLD_VI if is_vi else _JSON_SYSTEM_SCAFFOLD_EN
    # Use plain replacement so JSON braces in the scaffold are preserved literally.
    return (
        template
        .replace("{label_union}", "|".join(labels))
        .replace("{label_rules}", _json_label_rules(dataset, is_vi))
    )


def _is_vi(dataset: str, language: str) -> bool:
    return dataset == "vcom-data" or language == "vi"


def _base_instruction(is_vi: bool) -> str:
    return _BASE_INSTRUCTION_VI if is_vi else _BASE_INSTRUCTION_EN


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


def _json_strategy_note(is_vi: bool, strategy: str) -> str:
    if strategy != "cot":
        return ""
    if is_vi:
        return (
            "Hãy suy luận nội bộ theo từng bước nhưng không được in ra chain-of-thought. "
            "Chỉ xuất JSON cuối cùng đúng schema đã quy định."
        )
    return (
        "Reason internally step by step, but never reveal chain-of-thought. "
        "Only output final JSON following the required schema."
    )


def _user_contract(is_vi: bool) -> str:
    return _USER_CONTRACT_VI if is_vi else _USER_CONTRACT_EN


def _zero_shot_user_en(sentence: str) -> str:
    return (
        f"{_user_contract(False)}\n"
        "Follow the instruction and infer the tuple(s) directly from the sentence.\n"
        f"Input sentence: {sentence}\n"
        "Output:"
    )


def _zero_shot_user_vi(sentence: str) -> str:
    return (
        f"{_user_contract(True)}\n"
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
        "Sentence: The SD800 does at least as good a job as the SD700 in terms of color reproduction, sharpness, color saturation, detail retention, and crispness of the photos.\n"
        "Output: ([S] SD800 [O] SD700 [A] color reproduction [P] as good [L] Equal) ; ([S] SD800 [O] SD700 [A] sharpness [P] as good [L] Equal) ; ([S] SD800 [O] SD700 [A] color saturation [P] as good [L] Equal) ; ([S] SD800 [O] SD700 [A] detail retention [P] as good [L] Equal) ; ([S] SD800 [O] SD700 [A] crispness [P] as good [L] Equal)\n\n"
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


def _few_shot_examples_text(is_vi: bool) -> str:
    if is_vi:
        return (
            "Hãy học quy tắc trích xuất và định dạng đầu ra từ các ví dụ sau, sau đó áp dụng cho câu văn cuối cùng:\n"
            "Câu: Tương tự, thì ống kính góc rộng không có quá nhiều sự khác biệt so với ống kính chính.\n"
            "Kết quả: ([S] ống kính góc rộng [O] ống kính chính [A] [UNK] [P] không có quá nhiều sự khác biệt [L] EQL)\n\n"
            "Câu: Bên cạnh đó, iPhone 14 được nâng cấp bộ nhớ lên đến 6GB RAM cao hơn iPhone 13 đến 2GB RAM, cho khả năng đa nhiệm tốt hơn.\n"
            "Kết quả: ([S] iPhone 14 [O] iPhone 13 [A] bộ nhớ [P] cao hơn [L] COM+) ; ([S] iPhone 14 [O] iPhone 13 [A] khả năng đa nhiệm [P] tốt hơn [L] COM+)\n\n"
            "Câu: Bạn có thể selfie và sử dụng ở bể bơi mà không hề sợ bị hỏng máy.\n"
            "Kết quả: ([S] [UNK] [O] [UNK] [A] [UNK] [P] [UNK] [L] [UNK])"
        )
    return (
        "Learn the extraction patterns from the following examples and apply them to the final sentence:\n"
        "Sentence: The file-size gets even bigger if you shoot in RAW format instead of JPEG format.\n"
        "Output: ([S] RAW format [O] JPEG format [A] file-size [P] bigger [L] Better)\n\n"
        "Sentence: The SD800 does at least as good a job as the SD700 in terms of color reproduction, sharpness, color saturation, detail retention, and crispness of the photos.\n"
        "Output: ([S] SD800 [O] SD700 [A] color reproduction [P] as good [L] Equal) ; ([S] SD800 [O] SD700 [A] sharpness [P] as good [L] Equal) ; ([S] SD800 [O] SD700 [A] color saturation [P] as good [L] Equal) ; ([S] SD800 [O] SD700 [A] detail retention [P] as good [L] Equal) ; ([S] SD800 [O] SD700 [A] crispness [P] as good [L] Equal)\n\n"
        "Sentence: The pictures are truly professional quality.\n"
        "Output: ([S] [UNK] [O] [UNK] [A] [UNK] [P] [UNK] [L] [UNK])"
    )


def _few_shot_examples_json(is_vi: bool) -> str:
    if is_vi:
        return (
            "Hãy học quy tắc trích xuất và định dạng đầu ra JSON từ các ví dụ sau, sau đó áp dụng cho câu văn cuối cùng:\n"
            "Câu: Bên cạnh đó, iPhone 14 được nâng cấp bộ nhớ lên đến 6GB RAM cao hơn iPhone 13 đến 2GB RAM, cho khả năng đa nhiệm tốt hơn.\n"
            "Kết quả:\n"
            "{\n"
            "  \"comparisons\": [\n"
            "    {\n"
            "      \"label\": \"COM+\",\n"
            "      \"subject\": \"iPhone 14\",\n"
            "      \"object\": \"iPhone 13\",\n"
            "      \"aspect\": \"bộ nhớ\",\n"
            "      \"predicate\": \"cao hơn\"\n"
            "    },\n"
            "    {\n"
            "      \"label\": \"COM+\",\n"
            "      \"subject\": \"iPhone 14\",\n"
            "      \"object\": \"iPhone 13\",\n"
            "      \"aspect\": \"khả năng đa nhiệm\",\n"
            "      \"predicate\": \"tốt hơn\"\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Câu: Bạn có thể selfie và sử dụng ở bể bơi mà không hề sợ bị hỏng máy.\n"
            "Kết quả:\n"
            "{\n"
            "  \"comparisons\": []\n"
            "}"
        )
    return (
        "Learn the extraction patterns and JSON output format from the following examples, then apply them to the final sentence:\n"
        "Sentence: The file-size gets even bigger if you shoot in RAW format instead of JPEG format.\n"
        "Output:\n"
        "{\n"
        "  \"comparisons\": [\n"
        "    {\n"
        "      \"label\": \"Better\",\n"
        "      \"subject\": \"RAW format\",\n"
        "      \"object\": \"JPEG format\",\n"
        "      \"aspect\": \"file-size\",\n"
        "      \"predicate\": \"bigger\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Sentence: The SD800 does at least as good a job as the SD700 in terms of color reproduction, sharpness, color saturation, detail retention, and crispness of the photos.\n"
        "Output:\n"
        "{\n"
        "  \"comparisons\": [\n"
        "    {\n"
        "      \"label\": \"Equal\",\n"
        "      \"subject\": \"SD800\",\n"
        "      \"object\": \"SD700\",\n"
        "      \"aspect\": \"color reproduction\",\n"
        "      \"predicate\": \"as good\"\n"
        "    },\n"
        "    {\n"
        "      \"label\": \"Equal\",\n"
        "      \"subject\": \"SD800\",\n"
        "      \"object\": \"SD700\",\n"
        "      \"aspect\": \"sharpness\",\n"
        "      \"predicate\": \"as good\"\n"
        "    },\n"
        "    {\n"
        "      \"label\": \"Equal\",\n"
        "      \"subject\": \"SD800\",\n"
        "      \"object\": \"SD700\",\n"
        "      \"aspect\": \"color saturation\",\n"
        "      \"predicate\": \"as good\"\n"
        "    },\n"
        "    {\n"
        "      \"label\": \"Equal\",\n"
        "      \"subject\": \"SD800\",\n"
        "      \"object\": \"SD700\",\n"
        "      \"aspect\": \"detail retention\",\n"
        "      \"predicate\": \"as good\"\n"
        "    },\n"
        "    {\n"
        "      \"label\": \"Equal\",\n"
        "      \"subject\": \"SD800\",\n"
        "      \"object\": \"SD700\",\n"
        "      \"aspect\": \"crispness\",\n"
        "      \"predicate\": \"as good\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Sentence: The pictures are truly professional quality.\n"
        "Output:\n"
        "{\n"
        "  \"comparisons\": []\n"
        "}"
    )


def _short_user_prompt(sentence: str, is_vi: bool) -> str:
    tmpl = _SHORT_USER_VI if is_vi else _SHORT_USER_EN
    return tmpl.format(sentence=sentence)


def _short_user_json(sentence: str, is_vi: bool) -> str:
    tmpl = _SHORT_USER_JSON_VI if is_vi else _SHORT_USER_JSON_EN
    return tmpl.format(sentence=sentence)


def _zero_shot_user_en_json(sentence: str) -> str:
    return (
        f"{_JSON_SCHEMA_NOTE_EN}\n\n"
        "Extract all comparative quintuples from the input sentence and return JSON only.\n\n"
        f"Input sentence: {sentence}"
    )


def _zero_shot_user_vi_json(sentence: str) -> str:
    return (
        f"{_JSON_SCHEMA_NOTE_VI}\n\n"
        "Hãy trích xuất tất cả quintuple so sánh từ câu đầu vào và chỉ trả về JSON.\n\n"
        f"Câu đầu vào: {sentence}"
    )


def _few_shot_user_en_json(sentence: str) -> str:
    return (
        f"{_JSON_SCHEMA_NOTE_EN}\n\n"
        "Learn from examples, then solve the final sentence.\n\n"
        "Example 1\n"
        "Sentence: The file-size gets even bigger if you shoot in RAW format instead of JPEG format.\n"
        "Output:\n"
        "{\n"
        "  \"comparisons\": [\n"
        "    {\n"
        "      \"label\": \"Better\",\n"
        "      \"subject\": \"RAW format\",\n"
        "      \"object\": \"JPEG format\",\n"
        "      \"aspect\": \"file-size\",\n"
        "      \"predicate\": \"bigger\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Example 2\n"
        "Sentence: The SD800 does at least as good a job as the SD700 in terms of color reproduction, sharpness, color saturation, detail retention, and crispness of the photos.\n"
        "Output:\n"
        "{\n"
        "  \"comparisons\": [\n"
        "    {\n"
        "      \"label\": \"Equal\",\n"
        "      \"subject\": \"SD800\",\n"
        "      \"object\": \"SD700\",\n"
        "      \"aspect\": \"color reproduction\",\n"
        "      \"predicate\": \"as good\"\n"
        "    },\n"
        "    {\n"
        "      \"label\": \"Equal\",\n"
        "      \"subject\": \"SD800\",\n"
        "      \"object\": \"SD700\",\n"
        "      \"aspect\": \"sharpness\",\n"
        "      \"predicate\": \"as good\"\n"
        "    },\n"
        "    {\n"
        "      \"label\": \"Equal\",\n"
        "      \"subject\": \"SD800\",\n"
        "      \"object\": \"SD700\",\n"
        "      \"aspect\": \"color saturation\",\n"
        "      \"predicate\": \"as good\"\n"
        "    },\n"
        "    {\n"
        "      \"label\": \"Equal\",\n"
        "      \"subject\": \"SD800\",\n"
        "      \"object\": \"SD700\",\n"
        "      \"aspect\": \"detail retention\",\n"
        "      \"predicate\": \"as good\"\n"
        "    },\n"
        "    {\n"
        "      \"label\": \"Equal\",\n"
        "      \"subject\": \"SD800\",\n"
        "      \"object\": \"SD700\",\n"
        "      \"aspect\": \"crispness\",\n"
        "      \"predicate\": \"as good\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Example 3\n"
        "Sentence: The pictures are truly professional quality.\n"
        "Output:\n"
        "{\n"
        "  \"comparisons\": []\n"
        "}\n\n"
        f"Sentence: {sentence}"
    )


def _few_shot_user_vi_json(sentence: str) -> str:
    return (
        f"{_JSON_SCHEMA_NOTE_VI}\n\n"
        "Hãy học theo ví dụ rồi xử lý câu cuối cùng.\n\n"
        "Ví dụ 1\n"
        "Câu: Bên cạnh đó, iPhone 14 được nâng cấp bộ nhớ lên đến 6GB RAM cao hơn iPhone 13 đến 2GB RAM, cho khả năng đa nhiệm tốt hơn.\n"
        "Kết quả:\n"
        "{\n"
        "  \"comparisons\": [\n"
        "    {\n"
        "      \"label\": \"COM+\",\n"
        "      \"subject\": \"iPhone 14\",\n"
        "      \"object\": \"iPhone 13\",\n"
        "      \"aspect\": \"bộ nhớ\",\n"
        "      \"predicate\": \"cao hơn\"\n"
        "    },\n"
        "    {\n"
        "      \"label\": \"COM+\",\n"
        "      \"subject\": \"iPhone 14\",\n"
        "      \"object\": \"iPhone 13\",\n"
        "      \"aspect\": \"khả năng đa nhiệm\",\n"
        "      \"predicate\": \"tốt hơn\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Ví dụ 2\n"
        "Câu: Bạn có thể selfie và sử dụng ở bể bơi mà không hề sợ bị hỏng máy.\n"
        "Kết quả:\n"
        "{\n"
        "  \"comparisons\": []\n"
        "}\n\n"
        f"Câu: {sentence}"
    )


def _cot_user_en_json(sentence: str) -> str:
    return (
        f"{_JSON_SCHEMA_NOTE_EN}\n\n"
        "Reason internally in this order: S, O, A, P, L. Do not reveal reasoning. Return JSON only.\n\n"
        f"Input sentence: {sentence}"
    )


def _cot_user_vi_json(sentence: str) -> str:
    return (
        f"{_JSON_SCHEMA_NOTE_VI}\n\n"
        "Suy luận nội bộ theo thứ tự S, O, A, P, L. Không in lập luận. Chỉ trả về JSON.\n\n"
        f"Câu: {sentence}"
    )


def build_messages(
    sentence: str,
    language: str = "auto",
    dataset: str = "",
    strategy: str = "zero-shot",
    output_format: str = "json",
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
    if output_format not in {"json", "textual"}:
        raise ValueError("output_format must be one of: json, textual")

    if output_format == "json":
        extra = _few_shot_examples_json(is_vi) if strategy == "few-shot" else ""
        system_content = "\n\n".join(
            p
            for p in [
                _json_system_scaffold(dataset, is_vi),
                _json_strategy_note(is_vi, strategy),
            ]
            if p
        )

        if strategy == "few-shot" and extra:
            user_content = f"{extra}\n\n{_short_user_json(sentence, is_vi)}"
        else:
            user_content = _short_user_json(sentence, is_vi)

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    extra = _few_shot_examples_text(is_vi) if strategy == "few-shot" else ""
    system_content = "\n".join(
        p
        for p in [
            _base_instruction(is_vi),
            _user_contract(is_vi),
            _strategy_note(is_vi, strategy),
        ]
        if p
    )
    if strategy == "few-shot" and extra:
        user_content = f"{extra}\n\n{_short_user_prompt(sentence, is_vi)}"
    else:
        user_content = _short_user_prompt(sentence, is_vi)

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]
