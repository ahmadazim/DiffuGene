from __future__ import annotations

from typing import Dict, List, Optional, Any
import math


def _pow2_options(
    total_tokens: int,
    min_tokens: int = 1,
    max_tokens: Optional[int] = None,
) -> List[int]:
    """
    Power-of-two token options within [min_tokens, max_tokens], capped by total_tokens.
    Deterministic ordering: increasing.
    """
    T = int(total_tokens)
    if T <= 0:
        raise ValueError("total_tokens must be > 0")
    lo = max(1, int(min_tokens))
    hi = int(max_tokens) if max_tokens is not None else T
    if hi < lo:
        raise ValueError("max_tokens must be >= min_tokens")
    hi = min(hi, T)
    out: List[int] = []
    k = 0
    while (1 << k) <= hi:
        v = 1 << k
        if v >= lo:
            out.append(int(v))
        k += 1
    if not out:
        raise ValueError("No valid power-of-two options available.")
    return out


def solve_token_allocation_milp(
    w: List[float],
    total_tokens: int = 4096,
    min_tokens: int = 1,
    max_tokens: Optional[int] = None,
    solver_name: Optional[str] = None,
    verbose: bool = False,
    **_: Any,
) -> Dict:
    """
    Pure-Python, deterministic, exact allocator.

    Problem:
      minimize   sum_i |a_i - t_i|
      subject to a_i in {powers of 2 within [min_tokens, max_tokens]}
                 sum_i a_i = total_tokens

    where t_i = total_tokens * w_i / sum_j w_j.

    Notes:
      - `solver_name`, `verbose` are accepted for API compatibility but ignored.
    """
    if not isinstance(w, list) or len(w) == 0:
        raise ValueError("w must be a non-empty list of weights.")
    weights = [float(x) for x in w]
    if any(x < 0.0 for x in weights):
        raise ValueError("weights must be nonnegative.")
    sum_w = float(sum(weights))
    if not (sum_w > 0.0):
        raise ValueError("sum(w) must be > 0.")

    n = int(len(weights))
    T = int(total_tokens)
    if T <= 0:
        raise ValueError("total_tokens must be > 0.")

    options = _pow2_options(total_tokens=T, min_tokens=min_tokens, max_tokens=max_tokens)
    min_opt = min(options)
    max_opt = max(options)
    if n * min_opt > T or n * max_opt < T:
        raise ValueError("Infeasible: bounds/options cannot satisfy exact total_tokens.")

    targets: List[float] = [T * (wi / sum_w) for wi in weights]

    # DP over (i, s): best objective using first i items summing to s.
    inf = math.inf
    dp_prev = [inf] * (T + 1)
    dp_prev[0] = 0.0
    # backpointers: for each i and s, store (prev_s, chosen_opt).
    prev_s: List[List[int]] = [[-1] * (T + 1) for _ in range(n + 1)]
    prev_o: List[List[int]] = [[-1] * (T + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        tgt = float(targets[i - 1])
        dp_cur = [inf] * (T + 1)
        for s in range(0, T + 1):
            best_val = inf
            best_ps = -1
            best_opt = -1
            for opt in options:
                p = s - int(opt)
                if p < 0:
                    continue
                base = dp_prev[p]
                if base == inf:
                    continue
                cand = base + abs(float(opt) - tgt)
                # Deterministic tie-break: smaller objective, then smaller opt, then smaller prev sum.
                if (cand < best_val) or (
                    cand == best_val and (best_opt < 0 or int(opt) < best_opt)
                ) or (
                    cand == best_val and int(opt) == best_opt and (best_ps < 0 or p < best_ps)
                ):
                    best_val = cand
                    best_ps = p
                    best_opt = int(opt)
            dp_cur[s] = best_val
            prev_s[i][s] = best_ps
            prev_o[i][s] = best_opt
        dp_prev = dp_cur
    if dp_prev[T] == inf:
        raise RuntimeError("DP failed to find a feasible exact allocation.")
    allocations: List[int] = [0] * n
    s = T
    for i in range(n, 0, -1):
        opt = int(prev_o[i][s])
        ps = int(prev_s[i][s])
        if opt <= 0 or ps < 0:
            raise RuntimeError("DP backtracking failed.")
        allocations[i - 1] = opt
        s = ps
    if s != 0:
        raise RuntimeError("DP backtracking ended with non-zero remainder.")

    assignments: List[Dict[str, Any]] = []
    offsets: List[Dict[str, int]] = []
    cursor = 0
    abs_errors: List[float] = []
    for i, tok in enumerate(allocations):
        tgt = float(targets[i])
        ae = float(abs(float(tok) - tgt))
        abs_errors.append(ae)
        assignments.append({"i": int(i), "tokens": int(tok), "target": tgt, "abs_error": ae})
        offsets.append({"i": int(i), "token_start": int(cursor), "token_end": int(cursor + tok)})
        cursor += int(tok)
    return {
        "status": "Optimal",
        "objective": float(dp_prev[T]),
        "total_tokens": int(T),
        "allocations": allocations,
        "targets": targets,
        "abs_errors": abs_errors,
        "assignments": assignments,
        "offsets": offsets,
        "options": options,
        "sum_allocations": int(sum(allocations)),
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
