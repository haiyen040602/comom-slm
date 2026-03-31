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
) -> Dict[str, Dict[str, float]]:
    """Compute all metrics following the naming convention:

    {Matching Strategy}-{Level of Evaluation}-{Indication}

    where each dict value contains P/R/F1/support.

    Args:
        label_order: Fixed label sequence for T5 macro averaging.  Pass
            VCOM_LABEL_ORDER or CAMERA_COQE_LABEL_ORDER explicitly, or
            leave as None to auto-detect from the gold data.
    """
    # CEE: E/P/B x (S,O,A,P)
    cee: Dict[str, Dict[str, _Acc]] = {
        s: {e: _Acc() for e in CEE_ELEMS} for s in ("E", "P", "B")
    }

    # T4: E/B
    t4: Dict[str, _Acc] = {s: _Acc() for s in ("E", "B")}

    # T5: E/B x label
    t5: Dict[str, Dict[str, _Acc]] = {s: defaultdict(_Acc) for s in ("E", "B")}

    # Collect gold labels in first-seen order for auto-detection.
    _seen_labels: List[str] = []
    _seen_labels_set: Set[str] = set()

    for pred_str, gold_str in zip(predictions, gold_labels):
        pred_tuples = [t for t in _parse_tuples(pred_str) if not _is_all_unk(t)]
        gold_tuples = [t for t in _parse_tuples(gold_str) if not _is_all_unk(t)]

        # CEE
        for idx, elem in enumerate(CEE_ELEMS):
            g_vals = [gt[idx] for gt in gold_tuples]
            p_vals = [pt[idx] for pt in pred_tuples]

            for strat in ("E", "P", "B"):
                acc = cee[strat][elem]
                acc.gold += len(g_vals)
                acc.pred += len(p_vals)

                used: Set[int] = set()
                for pv in p_vals:
                    if strat == "E":
                        for gi, gv in enumerate(g_vals):
                            if gi not in used and _exact_match(pv, gv):
                                acc.tp += 1
                                acc.tp_prop += 1
                                used.add(gi)
                                break
                    elif strat == "B":
                        for gi, gv in enumerate(g_vals):
                            if gi not in used and _binary_match(pv, gv):
                                acc.tp += 1
                                used.add(gi)
                                break
                    else:  # proportional
                        best_gi = -1
                        best_sc = 0.0
                        for gi, gv in enumerate(g_vals):
                            if gi not in used:
                                sc = _proportional_score(pv, gv)
                                if sc > best_sc:
                                    best_sc = sc
                                    best_gi = gi
                        if best_gi >= 0:
                            acc.tp_prop += best_sc
                            used.add(best_gi)

        # T4 and T5
        for strat in ("E", "B"):
            match_fn = _exact_match if strat == "E" else _binary_match

            # T4
            acc4 = t4[strat]
            acc4.gold += len(gold_tuples)
            acc4.pred += len(pred_tuples)
            used4: Set[int] = set()
            for pt in pred_tuples:
                for gi, gt in enumerate(gold_tuples):
                    if gi not in used4 and all(match_fn(pt[i], gt[i]) for i in range(4)):
                        acc4.tp += 1
                        used4.add(gi)
                        break

            # T5 by label (from gold label space only)
            gold_labels_present = {gt[_L] for gt in gold_tuples}
            for lbl in gold_labels_present:
                if lbl and lbl not in _seen_labels_set:
                    _seen_labels.append(lbl)
                    _seen_labels_set.add(lbl)
                acc5 = t5[strat][lbl]
                g5 = [gt for gt in gold_tuples if gt[_L] == lbl]
                p5 = [pt for pt in pred_tuples if pt[_L] == lbl]
                acc5.gold += len(g5)
                acc5.pred += len(p5)

                used5: Set[int] = set()
                for pt in p5:
                    for gi, gt in enumerate(g5):
                        if gi not in used5 and all(match_fn(pt[i], gt[i]) for i in range(5)):
                            acc5.tp += 1
                            used5.add(gi)
                            break

    # Resolve label order for T5 output.
    _resolved_labels: Tuple[str, ...] = (
        label_order if label_order is not None
        else (tuple(_seen_labels) if _seen_labels else T5_LABEL_ORDER)
    )

    out: Dict[str, Dict[str, float]] = {}

    # CEE output
    for strat in ("E", "P", "B"):
        mode = "prop" if strat == "P" else "exact"

        for elem in CEE_ELEMS:
            out[f"{strat}-CEE-{elem}"] = cee[strat][elem].prf(mode)

        micro = _Acc()
        for elem in CEE_ELEMS:
            a = cee[strat][elem]
            micro.tp += a.tp
            micro.tp_prop += a.tp_prop
            micro.pred += a.pred
            micro.gold += a.gold
        out[f"{strat}-CEE-MICRO"] = micro.prf(mode)

        out[f"{strat}-CEE-MACRO"] = _macro_avg([cee[strat][e].prf(mode) for e in CEE_ELEMS])

    # T4 output
    for strat in ("E", "B"):
        out[f"{strat}-T4"] = t4[strat].prf("exact")

    # T5 output using the resolved label order.
    for strat in ("E", "B"):
        per_label_scores: List[Dict[str, float]] = []

        for lbl in _resolved_labels:
            a = t5[strat].get(lbl, _Acc())
            s = a.prf("exact")
            out[f"{strat}-T5-{lbl}"] = s
            per_label_scores.append(s)

        # Micro over resolved labels
        micro5 = _Acc()
        for lbl in _resolved_labels:
            a = t5[strat].get(lbl, _Acc())
            micro5.tp += a.tp
            micro5.pred += a.pred
            micro5.gold += a.gold
        out[f"{strat}-T5-MICRO"] = micro5.prf("exact")

        # Macro over resolved labels
        out[f"{strat}-T5-MACRO"] = _macro_avg(per_label_scores)

    return out


LEADERBOARD_KEYS = [
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
