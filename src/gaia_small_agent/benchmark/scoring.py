from __future__ import annotations

import re
import string


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _normalize_number_str(value: str) -> float:
    text = str(value)
    for char in ("$", "%", ","):
        text = text.replace(char, "")
    try:
        return float(text)
    except ValueError:
        return float("inf")


def _normalize_str(value: str, *, remove_punct: bool = True) -> str:
    no_spaces = re.sub(r"\s", "", str(value))
    if remove_punct:
        no_spaces = no_spaces.translate(str.maketrans("", "", string.punctuation))
    return no_spaces.lower()


def gaia_score(model_answer: str | None, ground_truth: str) -> bool:
    """GAIA leaderboard quasi-exact-match semantics.

    Kept local to avoid importing leaderboard code at runtime. The branch order mirrors
    the official scorer: raw numeric ground truth first, then comma/semicolon lists,
    then normalized strings.
    """
    answer = "None" if model_answer is None else str(model_answer)
    ground_truth = str(ground_truth)

    if _is_float(ground_truth):
        return _normalize_number_str(answer) == float(ground_truth)

    if any(char in ground_truth for char in (",", ";")):
        splitter = re.compile(r"[,;]")
        gt_elems = splitter.split(ground_truth)
        ma_elems = splitter.split(answer)
        if len(gt_elems) != len(ma_elems):
            return False
        comparisons: list[bool] = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if _is_float(gt_elem):
                comparisons.append(_normalize_number_str(ma_elem) == float(gt_elem))
            else:
                comparisons.append(
                    _normalize_str(ma_elem, remove_punct=False)
                    == _normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    return _normalize_str(answer) == _normalize_str(ground_truth)
