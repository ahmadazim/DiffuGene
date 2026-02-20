from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pulp


def _pow2_options(total_tokens: int, min_tokens: int = 1, max_tokens: Optional[int] = None) -> List[int]:
    if total_tokens <= 0:
        raise ValueError("total_tokens must be > 0")
    min_tokens = max(1, int(min_tokens))
    max_tokens = int(max_tokens) if max_tokens is not None else int(total_tokens)
    if max_tokens < min_tokens:
        raise ValueError("max_tokens must be >= min_tokens")
    out: List[int] = []
    k = 0
    while (1 << k) <= max_tokens:
        v = 1 << k
        if v >= min_tokens:
            out.append(v)
        k += 1
    if not out:
        raise ValueError("No valid power-of-two allocation options available.")
    return out


def solve_token_allocation_milp(
    w: List[float],
    total_tokens: int = 4096,
    min_tokens: int = 1,
    max_tokens: Optional[int] = None,
    time_limit: Optional[int] = None,
    solver_name: str = "CBC",
    verbose: bool = False,
) -> Dict:
    """
    Allocate power-of-two token counts per chromosome by minimizing L1 error
    against weight-proportional targets, with exact total token budget.
    """
    w_arr = np.asarray(w, dtype=float)
    if w_arr.ndim != 1 or len(w_arr) == 0:
        raise ValueError("w must be a non-empty 1D list/array.")
    if np.any(w_arr < 0):
        raise ValueError("w must be nonnegative.")
    if float(w_arr.sum()) <= 0:
        raise ValueError("sum(w) must be > 0.")

    n = int(len(w_arr))
    total_tokens = int(total_tokens)
    if total_tokens <= 0:
        raise ValueError("total_tokens must be > 0.")

    options = _pow2_options(total_tokens=total_tokens, min_tokens=min_tokens, max_tokens=max_tokens)
    if n * min(options) > total_tokens:
        raise ValueError(
            f"Infeasible: n={n} chromosomes with min option {min(options)} exceeds total_tokens={total_tokens}."
        )
    if n * max(options) < total_tokens:
        raise ValueError(
            f"Infeasible: n={n} chromosomes with max option {max(options)} cannot reach total_tokens={total_tokens}."
        )

    w_arr = w_arr / float(w_arr.sum())
    targets = total_tokens * w_arr

    model = pulp.LpProblem("TokenAllocationMILP", pulp.LpMinimize)

    x = {}
    for i in range(n):
        for opt in options:
            x[(i, opt)] = pulp.LpVariable(f"x_{i}_{opt}", lowBound=0, upBound=1, cat=pulp.LpBinary)

    t = {i: pulp.LpVariable(f"t_{i}", lowBound=0, cat=pulp.LpContinuous) for i in range(n)}

    # one option per chromosome
    for i in range(n):
        model += pulp.lpSum(x[(i, opt)] for opt in options) == 1, f"one_option_{i}"

    # exact token budget
    model += (
        pulp.lpSum(opt * x[(i, opt)] for i in range(n) for opt in options) == total_tokens
    ), "exact_total_tokens"

    # L1 error linearization
    for i in range(n):
        alloc_i = pulp.lpSum(opt * x[(i, opt)] for opt in options)
        model += alloc_i - float(targets[i]) <= t[i], f"abs_pos_{i}"
        model += float(targets[i]) - alloc_i <= t[i], f"abs_neg_{i}"

    model += pulp.lpSum(t[i] for i in range(n)), "minimize_total_abs_error"

    solver_name = str(solver_name).upper()
    if solver_name == "CBC":
        solver = pulp.PULP_CBC_CMD(msg=verbose, timeLimit=time_limit)
    elif solver_name == "GUROBI":
        solver = pulp.GUROBI_CMD(msg=verbose, timeLimit=time_limit)
    elif solver_name == "GLPK":
        solver = pulp.GLPK_CMD(msg=verbose, options=["--tmlim", str(time_limit or 0)])
    elif solver_name == "SCIP":
        solver = pulp.SCIP_CMD(msg=verbose, timeLimit=time_limit)
    else:
        raise ValueError(f"Unknown solver '{solver_name}'.")

    status_code = model.solve(solver)
    status = pulp.LpStatus[status_code]
    if status not in {"Optimal", "Not Solved", "Undefined", "Infeasible", "Unbounded"}:
        raise RuntimeError(f"Unexpected solver status: {status}")

    allocations = np.zeros((n,), dtype=int)
    assignments = []
    for i in range(n):
        picked = None
        for opt in options:
            if pulp.value(x[(i, opt)]) > 0.5:
                picked = int(opt)
                break
        if picked is None:
            picked = int(options[0])
        allocations[i] = picked
        assignments.append(
            {
                "i": int(i),
                "tokens": int(picked),
                "target": float(targets[i]),
                "abs_error": float(abs(float(picked) - float(targets[i]))),
            }
        )

    total = int(allocations.sum())
    offsets = []
    cursor = 0
    for i in range(n):
        start = int(cursor)
        end = int(cursor + allocations[i])
        offsets.append({"i": int(i), "token_start": start, "token_end": end})
        cursor = end

    return {
        "status": status,
        "objective": float(pulp.value(model.objective)) if pulp.value(model.objective) is not None else float("nan"),
        "total_tokens": int(total_tokens),
        "allocations": allocations,
        "targets": targets,
        "abs_errors": np.abs(allocations.astype(float) - targets),
        "assignments": assignments,
        "offsets": offsets,
        "options": options,
        "sum_allocations": total,
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
