from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _pow2_options(total_tokens: int, min_tokens: int = 1, max_tokens: Optional[int] = None) -> List[int]:
    if total_tokens <= 0:
        raise ValueError("total_tokens must be > 0")
    lo = max(1, int(min_tokens))
    hi = int(max_tokens) if max_tokens is not None else int(total_tokens)
    if hi < lo:
        raise ValueError("max_tokens must be >= min_tokens")
    out: List[int] = []
    k = 0
    while (1 << k) <= hi:
        v = 1 << k
        if v >= lo:
            out.append(v)
        k += 1
    if not out:
        raise ValueError("No valid power-of-two options available.")
    return out


def _nearest_option_index(target: float, options: List[int]) -> int:
    best_idx = 0
    best_dist = float("inf")
    for idx, opt in enumerate(options):
        dist = abs(float(opt) - float(target))
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def solve_token_allocation_milp(
    w: List[float],
    total_tokens: int = 4096,
    min_tokens: int = 1,
    max_tokens: Optional[int] = None,
) -> Dict:
    """
    Simple pure-Python allocator:
      1) assign each chromosome to nearest power-of-two token count
      2) greedily adjust up/down to match exact total token budget
    """
    w_arr = np.asarray(w, dtype=float)
    if w_arr.ndim != 1 or len(w_arr) == 0:
        raise ValueError("w must be a non-empty 1D list/array.")
    if np.any(w_arr < 0) or float(w_arr.sum()) <= 0:
        raise ValueError("w must be nonnegative and sum(w) must be > 0.")

    n = int(len(w_arr))
    total_tokens = int(total_tokens)
    if total_tokens <= 0:
        raise ValueError("total_tokens must be > 0.")

    options = _pow2_options(total_tokens=total_tokens, min_tokens=min_tokens, max_tokens=max_tokens)
    if n * min(options) > total_tokens:
        raise ValueError("Infeasible: minimum per-chromosome tokens exceed total_tokens.")
    if n * max(options) < total_tokens:
        raise ValueError("Infeasible: maximum per-chromosome tokens cannot reach total_tokens.")

    targets = total_tokens * (w_arr / float(w_arr.sum()))
    option_index = np.asarray([_nearest_option_index(float(t), options) for t in targets], dtype=int)

    allocations = np.asarray([int(options[idx]) for idx in option_index], dtype=int)

    # Greedily repair to exact budget while minimally increasing L1 error each step.
    delta = int(total_tokens - int(allocations.sum()))
    while delta != 0:
        if delta > 0:
            best_i = -1
            best_gain = float("inf")
            for i in range(n):
                idx = int(option_index[i])
                if idx >= len(options) - 1:
                    continue
                old = float(allocations[i])
                new = float(options[idx + 1])
                gain = abs(new - float(targets[i])) - abs(old - float(targets[i]))
                if gain < best_gain:
                    best_gain = gain
                    best_i = i
            if best_i < 0:
                raise RuntimeError("Could not increase allocations to reach total_tokens.")
            old_val = int(allocations[best_i])
            option_index[best_i] += 1
            allocations[best_i] = int(options[int(option_index[best_i])])
            delta -= int(allocations[best_i] - old_val)
        else:
            best_i = -1
            best_gain = float("inf")
            for i in range(n):
                idx = int(option_index[i])
                if idx <= 0:
                    continue
                old = float(allocations[i])
                new = float(options[idx - 1])
                gain = abs(new - float(targets[i])) - abs(old - float(targets[i]))
                if gain < best_gain:
                    best_gain = gain
                    best_i = i
            if best_i < 0:
                raise RuntimeError("Could not decrease allocations to reach total_tokens.")
            old_val = int(allocations[best_i])
            option_index[best_i] -= 1
            allocations[best_i] = int(options[int(option_index[best_i])])
            delta += int(old_val - allocations[best_i])

    assignments = []
    offsets = []
    cursor = 0
    for i in range(n):
        tok = int(allocations[i])
        assignments.append(
            {
                "i": int(i),
                "tokens": tok,
                "target": float(targets[i]),
                "abs_error": float(abs(float(tok) - float(targets[i]))),
            }
        )
        offsets.append({"i": int(i), "token_start": int(cursor), "token_end": int(cursor + tok)})
        cursor += tok

    abs_errors = np.abs(allocations.astype(float) - targets)
    return {
        "status": "Optimal",
        "objective": float(abs_errors.sum()),
        "total_tokens": int(total_tokens),
        "allocations": allocations,
        "targets": targets,
        "abs_errors": abs_errors,
        "assignments": assignments,
        "offsets": offsets,
        "options": options,
        "sum_allocations": int(allocations.sum()),
    }


def organize_token_solution(result: Dict) -> Dict[int, Dict[str, int]]:
    """
    Return index -> token allocation metadata.
    """
    out: Dict[int, Dict[str, int]] = {}
    offsets = {int(o["i"]): o for o in result.get("offsets", [])}
    for a in result.get("assignments", []):
        i = int(a["i"])
        off = offsets.get(i, {"token_start": 0, "token_end": int(a["tokens"])})
        out[i] = {
            "tokens": int(a["tokens"]),
            "token_start": int(off["token_start"]),
            "token_end": int(off["token_end"]),
        }
    return out
