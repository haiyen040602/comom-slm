import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# Tuple format:
# ([S] ... [O] ... [A] ... [P] ... [L] ...)
_TUPLE_RE = re.compile(
    r"\[S\]\s*(.*?)\s*\[O\]\s*(.*?)\s*\[A\]\s*(.*?)\s*\[P\]\s*(.*?)\s*\[L\]\s*(.*?)(?=\)|\n|;|$)",
    re.DOTALL,
)

_S, _O, _A, _P, _L = 0, 1, 2, 3, 4
CEE_ELEMS = ("S", "O", "A", "P")

# Label orders for T5 macro averaging — must match the dataset's label convention.
VCOM_LABEL_ORDER = ("EQL", "DIF", "COM", "COM+", "COM-", "SUP", "SUP+", "SUP-")
CAMERA_COQE_LABEL_ORDER = ("Better", "Worse", "Equal", "Different")

# Backward-compat alias (default when no label_order is passed).
T5_LABEL_ORDER = VCOM_LABEL_ORDER


class _Acc:
    __slots__ = ("tp", "tp_prop", "pred", "gold")

    def __init__(self) -> None:
        self.tp: float = 0.0
        self.tp_prop: float = 0.0
        self.pred: int = 0
        self.gold: int = 0

    def prf(self, mode: str = "exact") -> Dict[str, float]:
        tp = self.tp_prop if mode == "prop" else self.tp
        p = tp / self.pred if self.pred > 0 else 0.0
        r = tp / self.gold if self.gold > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"P": p, "R": r, "F1": f1, "support": float(self.gold)}


def _parse_tuples(text: str) -> List[Tuple[str, str, str, str, str]]:
    tuples: List[Tuple[str, str, str, str, str]] = []
    for part in (text or "").split(";"):
        m = _TUPLE_RE.search(part.strip().strip("()"))
        if m:
            tuples.append(tuple(v.strip() for v in m.groups()))  # type: ignore[arg-type]
    return tuples


def _is_all_unk(t: Tuple[str, str, str, str, str]) -> bool:
    return all((s or "").strip() == "[UNK]" for s in t)


def _normalise(s: str) -> str:
    return " ".join(s.lower().split())


def _tokens(s: str) -> List[str]:
    return _normalise(s).split()


def _exact_match(pred: str, gold: str) -> bool:
    return _normalise(pred) == _normalise(gold)


def _binary_match(pred: str, gold: str) -> bool:
    return bool(set(_tokens(gold)) & set(_tokens(pred)))


def _proportional_score(pred: str, gold: str) -> float:
    gt = _tokens(gold)
    pd = _tokens(pred)
    if not gt:
        return 1.0 if not pd else 0.0
    pd_set = set(pd)
    overlap = sum(1 for w in gt if w in pd_set)
    return overlap / len(gt)


def _macro_avg(scores: List[Dict[str, float]]) -> Dict[str, float]:
    if not scores:
        return {"P": 0.0, "R": 0.0, "F1": 0.0, "support": 0.0}
    n = len(scores)
    return {
        "P": sum(s["P"] for s in scores) / n,
        "R": sum(s["R"] for s in scores) / n,
        "F1": sum(s["F1"] for s in scores) / n,
        "support": sum(s.get("support", 0.0) for s in scores),
    }


def compute_coqe_metrics(
    predictions: List[str],
    gold_labels: List[str],
    label_order: Optional[Tuple[str, ...]] = None,
    match_mode: str = "index-match",
) -> Dict[str, Dict[str, float]]:
    """Compute metrics with counting rules aligned to evaluate_v1.py.

    Output keys are preserved as:
      {E|P|B}-CEE-{S|O|A|P|MICRO|MACRO}
      {E|B}-T4
      {E|B}-T5-{label|MICRO|MACRO}
            SENT-CMP

        Notes:
            - match_mode='index-match': compare index spans when available (evaluate_v1-style).
            - match_mode='non-index-match': compare phrase tokens only.
            - TP/FP/FN accumulation rules follow evaluate_v1.
    """

    if match_mode not in {"index-match", "non-index-match"}:
        raise ValueError("match_mode must be one of: 'index-match', 'non-index-match'")
    if len(predictions) != len(gold_labels):
        raise ValueError(
            f"Sample count mismatch: predictions={len(predictions)} vs gold_labels={len(gold_labels)}"
        )

    def _prf_from_counts(tp: float, fp: float, fn: float) -> Dict[str, float]:
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"P": p, "R": r, "F1": f1, "support": tp + fn}

    def _slot_set(text: str) -> Set[str]:
        t = (text or "").strip()
        if t == "" or t == "[UNK]":
            return set()

        def _slot_word_tokens(value: str) -> List[str]:
            words: List[str] = []
            for piece in value.split():
                if "&&" in piece:
                    _, right = piece.split("&&", 1)
                    right = right.strip()
                    if right:
                        words.append(_normalise(right))
                else:
                    norm = _normalise(piece)
                    if norm:
                        words.append(norm)
            return [w for w in words if w]

        if match_mode == "non-index-match":
            return set(_slot_word_tokens(t))

        idxs: Set[str] = set()
        for piece in t.split():
            left = piece.split("&&", 1)[0].strip()
            if left.isdigit():
                idxs.add(left)
        if idxs:
            return idxs
        return set(_slot_word_tokens(t))

    def _tuple_obj(t: Tuple[str, str, str, str, str]) -> Dict[str, object]:
        return {
            "S": _slot_set(t[_S]),
            "O": _slot_set(t[_O]),
            "A": _slot_set(t[_A]),
            "P": _slot_set(t[_P]),
            "L": (t[_L] or "").strip(),
            "raw": t,
        }

    def _tuple_dedup(items: List[Tuple[str, str, str, str, str]]) -> List[Tuple[str, str, str, str, str]]:
        seen: Set[Tuple[str, str, str, str, str]] = set()
        out: List[Tuple[str, str, str, str, str]] = []
        for t in items:
            k = tuple(_normalise(x) for x in t)
            if k not in seen:
                seen.add(k)
                out.append(t)
        return out

    def _entity_exact(e1: Set[str], e2: Set[str]) -> int:
        return int(e1 == e2)

    def _entity_binary(e1: Set[str], e2: Set[str]) -> int:
        return int(len(e1.intersection(e2)) != 0)

    def _entity_prop(e1: Set[str], e2: Set[str]) -> float:
        if not e1:
            return 0.0
        return len(e1.intersection(e2)) / len(e1)

    def _sentence_cee_counts(gold_entities: List[Set[str]], pred_entities: List[Set[str]]) -> Dict[str, float]:
        ret = {
            "E-TP": 0.0,
            "E-FP": 0.0,
            "E-FN": 0.0,
            "P-TP": 0.0,
            "P-FP": 0.0,
            "P-FN": 0.0,
            "B-TP": 0.0,
            "B-FP": 0.0,
            "B-FN": 0.0,
        }

        for pred in pred_entities:
            if any(_entity_exact(pred, g) == 1 for g in gold_entities):
                ret["E-TP"] += 1
            else:
                ret["E-FP"] += 1

            max_match = max((_entity_prop(pred, g) for g in gold_entities), default=0.0)
            ret["P-TP"] += max_match
            ret["P-FP"] += (1 - max_match)

            if any(_entity_binary(pred, g) == 1 for g in gold_entities):
                ret["B-TP"] += 1
            else:
                ret["B-FP"] += 1

        for gold in gold_entities:
            if not any(_entity_exact(gold, p) == 1 for p in pred_entities):
                ret["E-FN"] += 1

            max_match = max((_entity_prop(gold, p) for p in pred_entities), default=0.0)
            ret["P-FN"] += (1 - max_match)

            if not any(_entity_binary(gold, p) == 1 for p in pred_entities):
                ret["B-FN"] += 1

        return ret

    def _tuple_exact(t1: Dict[str, object], t2: Dict[str, object], omit_label: bool) -> int:
        elems_equal = all(t1[k] == t2[k] for k in CEE_ELEMS)
        label_equal = True if omit_label else (t1["L"] == t2["L"])
        return int(elems_equal and label_equal)

    def _tuple_binary(t1: Dict[str, object], t2: Dict[str, object], omit_label: bool) -> int:
        elems_ok = all(
            (len(t1[k]) == 0 and len(t2[k]) == 0) or len(t1[k].intersection(t2[k])) != 0  # type: ignore[arg-type]
            for k in CEE_ELEMS
        )
        label_ok = True if omit_label else (t1["L"] == t2["L"])
        return int(elems_ok and label_ok)

    def _sentence_tuple_counts(
        gold_tuples: List[Dict[str, object]],
        pred_tuples: List[Dict[str, object]],
        omit_label: bool,
    ) -> Dict[str, float]:
        ret = {
            "E-TP": 0.0,
            "E-FP": 0.0,
            "E-FN": 0.0,
            "B-TP": 0.0,
            "B-FP": 0.0,
            "B-FN": 0.0,
        }

        for pred in pred_tuples:
            if any(_tuple_exact(pred, g, omit_label) == 1 for g in gold_tuples):
                ret["E-TP"] += 1
            else:
                ret["E-FP"] += 1

            if any(_tuple_binary(pred, g, omit_label) == 1 for g in gold_tuples):
                ret["B-TP"] += 1
            else:
                ret["B-FP"] += 1

        for gold in gold_tuples:
            if not any(_tuple_exact(gold, p, omit_label) == 1 for p in pred_tuples):
                ret["E-FN"] += 1

            if not any(_tuple_binary(gold, p, omit_label) == 1 for p in pred_tuples):
                ret["B-FN"] += 1

        return ret

    # Parse all sentence tuples first.
    sent_gold_raw: List[List[Tuple[str, str, str, str, str]]] = []
    sent_pred_raw: List[List[Tuple[str, str, str, str, str]]] = []

    seen_labels: List[str] = []
    seen_label_set: Set[str] = set()

    for pred_str, gold_str in zip(predictions, gold_labels):
        pred_t = [t for t in _parse_tuples(pred_str) if not _is_all_unk(t)]
        gold_t = [t for t in _parse_tuples(gold_str) if not _is_all_unk(t)]

        for gt in gold_t:
            lbl = (gt[_L] or "").strip()
            if lbl and lbl not in seen_label_set:
                seen_label_set.add(lbl)
                seen_labels.append(lbl)

        sent_pred_raw.append(pred_t)
        sent_gold_raw.append(gold_t)

    resolved_labels: Tuple[str, ...] = (
        label_order if label_order is not None else (tuple(seen_labels) if seen_labels else T5_LABEL_ORDER)
    )

    out: Dict[str, Dict[str, float]] = {}

    # ---- Sentence-level comparison detection ----
    # Positive class: sentence contains at least one non-[UNK] comparative tuple.
    sent_tp = 0.0
    sent_fp = 0.0
    sent_fn = 0.0
    for gold_tuples, pred_tuples in zip(sent_gold_raw, sent_pred_raw):
        gold_has_cmp = len(gold_tuples) > 0
        pred_has_cmp = len(pred_tuples) > 0
        if pred_has_cmp and gold_has_cmp:
            sent_tp += 1
        elif pred_has_cmp and (not gold_has_cmp):
            sent_fp += 1
        elif (not pred_has_cmp) and gold_has_cmp:
            sent_fn += 1

    out["SENT-CMP"] = _prf_from_counts(sent_tp, sent_fp, sent_fn)

    # ---- CEE (E/P/B, per element + micro + macro) ----
    cee_element_scores: Dict[str, Dict[str, Dict[str, float]]] = {s: {} for s in ("E", "P", "B")}

    for elem_i, elem_k in enumerate(CEE_ELEMS):
        agg = {
            "E-TP": 0.0,
            "E-FP": 0.0,
            "E-FN": 0.0,
            "P-TP": 0.0,
            "P-FP": 0.0,
            "P-FN": 0.0,
            "B-TP": 0.0,
            "B-FP": 0.0,
            "B-FN": 0.0,
        }

        for gold_tuples, pred_tuples in zip(sent_gold_raw, sent_pred_raw):
            gold_entities = []
            pred_entities = []

            # evaluate_v1-style dedup at element level per sentence
            seen_g: Set[frozenset] = set()
            for t in gold_tuples:
                e = _slot_set(t[elem_i])
                if len(e) == 0:
                    continue
                fe = frozenset(e)
                if fe not in seen_g:
                    seen_g.add(fe)
                    gold_entities.append(e)

            seen_p: Set[frozenset] = set()
            for t in pred_tuples:
                e = _slot_set(t[elem_i])
                if len(e) == 0:
                    continue
                fe = frozenset(e)
                if fe not in seen_p:
                    seen_p.add(fe)
                    pred_entities.append(e)

            sent_counts = _sentence_cee_counts(gold_entities, pred_entities)
            for k, v in sent_counts.items():
                agg[k] += v

        e_score = _prf_from_counts(agg["E-TP"], agg["E-FP"], agg["E-FN"])
        p_score = _prf_from_counts(agg["P-TP"], agg["P-FP"], agg["P-FN"])
        b_score = _prf_from_counts(agg["B-TP"], agg["B-FP"], agg["B-FN"])

        out[f"E-CEE-{elem_k}"] = e_score
        out[f"P-CEE-{elem_k}"] = p_score
        out[f"B-CEE-{elem_k}"] = b_score

        cee_element_scores["E"][elem_k] = e_score
        cee_element_scores["P"][elem_k] = p_score
        cee_element_scores["B"][elem_k] = b_score

    for strat in ("E", "P", "B"):
        # micro over aggregated TP/FP/FN from per-element scores
        tp = fp = fn = 0.0
        for elem_k in CEE_ELEMS:
            s = cee_element_scores[strat][elem_k]
            # reconstruct counts using support and P/R when possible
            # better: recompute from metric identity with support = tp+fn
            # store micro by summing per-element supports and re-aggregating from P/R is lossy,
            # so compute directly from sentence-level aggregation already done above is preferable.
            # To keep exact evaluate_v1 behavior, rebuild from sentence totals:
            # we approximate through P/R/support here only for output cohesion.
            # Replace with exact counters by re-running sentence loops per strategy.
            pass

    # exact micro counters for CEE (evaluate_v1 style)
    for strat in ("E", "P", "B"):
        key_tp = f"{strat}-TP"
        key_fp = f"{strat}-FP"
        key_fn = f"{strat}-FN"
        micro = {key_tp: 0.0, key_fp: 0.0, key_fn: 0.0}

        for elem_i, _ in enumerate(CEE_ELEMS):
            for gold_tuples, pred_tuples in zip(sent_gold_raw, sent_pred_raw):
                gold_entities = []
                pred_entities = []

                seen_g: Set[frozenset] = set()
                for t in gold_tuples:
                    e = _slot_set(t[elem_i])
                    if len(e) == 0:
                        continue
                    fe = frozenset(e)
                    if fe not in seen_g:
                        seen_g.add(fe)
                        gold_entities.append(e)

                seen_p: Set[frozenset] = set()
                for t in pred_tuples:
                    e = _slot_set(t[elem_i])
                    if len(e) == 0:
                        continue
                    fe = frozenset(e)
                    if fe not in seen_p:
                        seen_p.add(fe)
                        pred_entities.append(e)

                c = _sentence_cee_counts(gold_entities, pred_entities)
                micro[key_tp] += c[key_tp]
                micro[key_fp] += c[key_fp]
                micro[key_fn] += c[key_fn]

        out[f"{strat}-CEE-MICRO"] = _prf_from_counts(micro[key_tp], micro[key_fp], micro[key_fn])

        macro_list = [out[f"{strat}-CEE-{e}"] for e in CEE_ELEMS]
        out[f"{strat}-CEE-MACRO"] = _macro_avg(macro_list)

    # ---- T4 (E/B) ----
    for strat in ("E", "B"):
        agg = {"E-TP": 0.0, "E-FP": 0.0, "E-FN": 0.0, "B-TP": 0.0, "B-FP": 0.0, "B-FN": 0.0}
        for gold_tuples, pred_tuples in zip(sent_gold_raw, sent_pred_raw):
            g_objs = [_tuple_obj(t) for t in gold_tuples]
            p_objs = [_tuple_obj(t) for t in pred_tuples]
            c = _sentence_tuple_counts(g_objs, p_objs, omit_label=True)
            for k in agg:
                agg[k] += c[k]

        out[f"{strat}-T4"] = _prf_from_counts(agg[f"{strat}-TP"], agg[f"{strat}-FP"], agg[f"{strat}-FN"])

    # ---- T5 per label (E/B), with evaluate_v1-style dedup per sentence+label ----
    label_scores_by_strat: Dict[str, List[Dict[str, float]]] = {"E": [], "B": []}

    for lbl in resolved_labels:
        per_sent_gold: List[List[Tuple[str, str, str, str, str]]] = []
        per_sent_pred: List[List[Tuple[str, str, str, str, str]]] = []

        for gold_tuples, pred_tuples in zip(sent_gold_raw, sent_pred_raw):
            g_lbl = _tuple_dedup([t for t in gold_tuples if (t[_L] or "").strip() == lbl])
            p_lbl = _tuple_dedup([t for t in pred_tuples if (t[_L] or "").strip() == lbl])
            per_sent_gold.append(g_lbl)
            per_sent_pred.append(p_lbl)

        for strat in ("E", "B"):
            agg = {"E-TP": 0.0, "E-FP": 0.0, "E-FN": 0.0, "B-TP": 0.0, "B-FP": 0.0, "B-FN": 0.0}
            for g_lbl, p_lbl in zip(per_sent_gold, per_sent_pred):
                g_objs = [_tuple_obj(t) for t in g_lbl]
                p_objs = [_tuple_obj(t) for t in p_lbl]
                c = _sentence_tuple_counts(g_objs, p_objs, omit_label=False)
                for k in agg:
                    agg[k] += c[k]

            s = _prf_from_counts(agg[f"{strat}-TP"], agg[f"{strat}-FP"], agg[f"{strat}-FN"])
            out[f"{strat}-T5-{lbl}"] = s
            label_scores_by_strat[strat].append(s)

    for strat in ("E", "B"):
        # micro over all labels
        agg = {"E-TP": 0.0, "E-FP": 0.0, "E-FN": 0.0, "B-TP": 0.0, "B-FP": 0.0, "B-FN": 0.0}
        for lbl in resolved_labels:
            per_sent_gold = []
            per_sent_pred = []
            for gold_tuples, pred_tuples in zip(sent_gold_raw, sent_pred_raw):
                g_lbl = _tuple_dedup([t for t in gold_tuples if (t[_L] or "").strip() == lbl])
                p_lbl = _tuple_dedup([t for t in pred_tuples if (t[_L] or "").strip() == lbl])
                per_sent_gold.append(g_lbl)
                per_sent_pred.append(p_lbl)

            for g_lbl, p_lbl in zip(per_sent_gold, per_sent_pred):
                g_objs = [_tuple_obj(t) for t in g_lbl]
                p_objs = [_tuple_obj(t) for t in p_lbl]
                c = _sentence_tuple_counts(g_objs, p_objs, omit_label=False)
                for k in agg:
                    agg[k] += c[k]

        out[f"{strat}-T5-MICRO"] = _prf_from_counts(agg[f"{strat}-TP"], agg[f"{strat}-FP"], agg[f"{strat}-FN"])
        out[f"{strat}-T5-MACRO"] = _macro_avg(label_scores_by_strat[strat])

    return out


LEADERBOARD_KEYS = [
    "SENT-CMP",
    "E-CEE-S",
    "E-CEE-O",
    "E-CEE-A",
    "E-CEE-P",
    "E-CEE-MICRO",
    "E-CEE-MACRO",
    "P-CEE-MICRO",
    "P-CEE-MACRO",
    "B-CEE-MICRO",
    "B-CEE-MACRO",
    "E-T4",
    "B-T4",
    "E-T5-MICRO",
    "E-T5-MACRO",
    "B-T5-MACRO",
]


def leaderboard_row(metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {f"{k}-F1": metrics[k]["F1"] for k in LEADERBOARD_KEYS if k in metrics}


def metrics_to_lines(metrics: Dict[str, Dict[str, float]]) -> List[str]:
    lines = ["Metric,P,R,F1,Support"]
    for key, val in sorted(metrics.items()):
        lines.append(
            f"{key},{val['P']:.4f},{val['R']:.4f},{val['F1']:.4f},{int(val.get('support', 0))}"
        )
    return lines


def print_metrics_table(
    metrics: Dict[str, Dict[str, float]],
    keys: Optional[List[str]] = None,
    title: str = "",
) -> None:
    keys = keys or [k for k in LEADERBOARD_KEYS if k in metrics]
    width = max((len(k) for k in keys), default=20) + 2
    header = f"  {'Metric':<{width}} {'P':>8} {'R':>8} {'F1':>8} {'Support':>8}"
    sep = "=" * len(header)
    if title:
        print(f"\n{sep}\n  {title}")
    print(sep)
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for k in keys:
        if k not in metrics:
            continue
        v = metrics[k]
        marker = "  <- RANKING" if k == "E-T5-MACRO" else ""
        print(
            f"  {k:<{width}} {v['P']:>8.4f} {v['R']:>8.4f} "
            f"{v['F1']:>8.4f} {int(v.get('support', 0)):>8}{marker}"
        )
    print(f"{sep}\n")
