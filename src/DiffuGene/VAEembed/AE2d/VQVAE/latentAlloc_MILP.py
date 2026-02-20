from __future__ import annotations
from dataclasses import dataclass
from math import ceil, log2
from typing import List, Dict, Tuple, Optional

import numpy as np
import pulp


@dataclass(frozen=True)
class QuadNode:
    """A node in the L-level quadtree over an M_eff x M_eff grid."""
    id: int
    depth: int                 # 0..L (root=0)
    x0: int                    # top-left x (integer pixel coordinate)
    y0: int                    # top-left y (integer pixel coordinate)
    size: int                  # side length in pixels (M_eff // 2**depth)
    parent: Optional[int]      # parent node id or None for root
    children: Tuple[int, ...]  # child ids (empty tuple for leaves)


def _next_power_of_two(n: int) -> int:
    """Small helper: pad n up to the next power of two (if not already)."""
    if n <= 0:
        raise ValueError("M must be positive")
    if (n & (n - 1)) == 0:
        return n
    return 1 << (int(log2(n - 1)) + 1)


def _build_quadtree(M_eff: int) -> Tuple[List[QuadNode], Dict[int, QuadNode]]:
    """Build complete quadtree nodes for an M_eff x M_eff grid (M_eff = 2^L)."""
    L = int(log2(M_eff))
    nodes: List[QuadNode] = []
    # BFS construction
    q = []
    root = QuadNode(id=0, depth=0, x0=0, y0=0, size=M_eff, parent=None, children=tuple())
    nodes.append(root)
    q.append(0)

    while q:
        nid = q.pop(0)
        node = nodes[nid]
        if node.depth == L:
            # leaf
            continue
        # split into 4 children
        half = node.size // 2
        base_id = len(nodes)
        ch = []
        # Children: (x,y) order: (x0,y0), (x0+half,y0), (x0,y0+half), (x0+half,y0+half)
        coords = [
            (node.x0, node.y0),
            (node.x0 + half, node.y0),
            (node.x0, node.y0 + half),
            (node.x0 + half, node.y0 + half),
        ]
        for (cx, cy) in coords:
            cid = len(nodes)
            child = QuadNode(
                id=cid,
                depth=node.depth + 1,
                x0=cx,
                y0=cy,
                size=half,
                parent=nid,
                children=tuple()
            )
            nodes.append(child)
            ch.append(cid)
            q.append(cid)
        # patch node with children (dataclasses are frozen => rebuild)
        nodes[nid] = QuadNode(
            id=node.id, depth=node.depth, x0=node.x0, y0=node.y0, size=node.size,
            parent=node.parent, children=tuple(ch)
        )

    id2node = {nd.id: nd for nd in nodes}
    return nodes, id2node


def solve_quadtree_milp(
    w: List[float],
    M: int,
    time_limit: Optional[int] = None,
    solver_name: str = "CBC",
    verbose: bool = False,
    stop_min_side: Optional[int] = None,
    stop_max_side: Optional[int] = None,
) -> Dict:
    """
    Solve the Quadtree MILP exactly as specified (with standard linearization for coverage).

    Parameters
    ----------
    w : list/array of floats (length N)
        Relative information weights, should sum to 1 (will be renormalized safely).
    M : int
        Requested grid side length. If not a power of two, we pad up to M_eff = 2^L.
    time_limit : int or None
        Optional solver time limit in seconds.
    solver_name : {"CBC","GUROBI","GLPK","SCIP"} (if installed in PuLP)
        MILP solver to use via PuLP.
    verbose : bool
        Whether to print solver logs.

    Returns
    -------
    result : dict
        {
          "status": <pulp status string>,
          "objective": float,
          "M_eff": int,
          "L": int,
          "areas": np.ndarray shape (N,),       # actual assigned areas a_i
          "targets": np.ndarray shape (N,),     # M_eff^2 * w_i (renormalized w)
          "abs_errors": np.ndarray shape (N,),  # |a_i - target_i|
          "assignments": List[dict],            # one per block i
              # dict with keys: i, node_id, x0, y0, size, depth, area
          "selected_nodes": List[int],          # node ids with s_q == 1
        }

    Raises
    ------
    ValueError
        If feasibility is impossible due to quadtree leaf-count arithmetic:
        N must satisfy N ≡ 1 (mod 3) and N ≤ 4^L.
    """
    # --- inputs & padding ---
    w = np.asarray(w, dtype=float)
    if w.ndim != 1 or len(w) == 0:
        raise ValueError("w must be a non-empty 1D vector")
    if np.any(w < 0):
        raise ValueError("w must be nonnegative")
    if M <= 0:
        raise ValueError("M must be positive")

    # renormalize w safely
    s = float(w.sum())
    if s <= 0:
        raise ValueError("sum(w) must be positive")
    w = w / s
    N = len(w)

    # pad M up to power-of-two as required
    M_eff = _next_power_of_two(M)
    L = int(log2(M_eff))
    total_area = M_eff * M_eff  # = 4^L

    # quadtree feasibility check: the number of stops (selected squares) equals number of blocks
    # and must be of the form 1 + 3k, with maximum 4^L.
    if N > total_area:
        raise ValueError(
            f"Infeasible: N={N} blocks but max number of unit leaves is 4^L={total_area}."
        )
    if (N - 1) % 3 != 0:
        raise ValueError(
            f"Infeasible: With exact tiling by dyadic squares and one block per square, "
            f"the number of selected squares must be 1 + 3k. Your N={N} "
            f"is not congruent to 1 mod 3."
        )

    # build full quadtree
    nodes, id2node = _build_quadtree(M_eff)
    all_ids = [nd.id for nd in nodes]
    root_id = 0
    internal_ids = [nd.id for nd in nodes if len(nd.children) > 0]
    leaf_ids = [nd.id for nd in nodes if len(nd.children) == 0]

    # precompute A(q) = 4^{ℓ(q)} with ℓ(q) = L - depth(q)
    A = {qid: 4 ** (L - id2node[qid].depth) for qid in all_ids}  # integer area in pixels

    # ----------------------------
    # Infer GLOBAL stop size bounds if not given
    # Rule:
    #   v_min = M_eff * sqrt(min(w));  use lower power of 2, but if within 5% *above* a power-of-two
    #           boundary, step DOWN one more level.
    #   v_max = M_eff * sqrt(max(w));  use higher power of 2, but if within 5% *below* a power-of-two
    #           boundary, step UP one more level.
    # Clamp to [1, M_eff] and ensure min <= max.
    # ----------------------------
    def _pow2_floor(x: float) -> int:
        return 1 << int(np.floor(np.log2(max(1.0, x))))
    def _pow2_ceil(x: float) -> int:
        return 1 << int(np.ceil(np.log2(max(1.0, x))))
    if stop_min_side is None or stop_max_side is None:
        wmin = float(w.min())
        wmax = float(w.max())
        v_min = M_eff * np.sqrt(wmin)  # desired side for smallest block
        v_max = M_eff * np.sqrt(wmax)  # desired side for largest block
        p_min = _pow2_floor(v_min)
        if v_min / p_min <= 1.05:
            p_min = max(1, p_min // 2)
        p_max = _pow2_ceil(v_max)
        if v_max >= 0.95 * p_max:
            p_max = min(M_eff, p_max * 2)
        if stop_min_side is None:
            stop_min_side = int(max(1, min(p_min, M_eff)))
        if stop_max_side is None:
            stop_max_side = int(max(1, min(p_max, M_eff)))
    stop_min_side = int(max(1, min(stop_min_side, M_eff))) if stop_min_side is not None else 1
    stop_max_side = int(max(1, min(stop_max_side, M_eff))) if stop_max_side is not None else M_eff
    if stop_min_side > stop_max_side:
        stop_min_side, stop_max_side = stop_max_side, stop_min_side
    print(f"Using global stop size bounds: stop_min_side: {stop_min_side}, stop_max_side: {stop_max_side}")

    # --- MILP model ---
    model = pulp.LpProblem("Dyadic_Treemap_Quadtree_MILP", pulp.LpMinimize)

    # decision variables
    s_var = pulp.LpVariable.dicts("s", all_ids, lowBound=0, upBound=1, cat=pulp.LpBinary)  # stop here?
    # auxiliary for exact-tiling coverage (standard linearization of "stop vs split")
    a_var = pulp.LpVariable.dicts("a", all_ids, lowBound=0, upBound=1, cat=pulp.LpBinary)  # active?
    y_var = pulp.LpVariable.dicts("y", internal_ids, lowBound=0, upBound=1, cat=pulp.LpBinary)  # split?

    x_var = {}
    for i in range(N):
        for q in all_ids:
            x_var[(i, q)] = pulp.LpVariable(f"x_{i}_{q}", lowBound=0, upBound=1, cat=pulp.LpBinary)

    t_var = {i: pulp.LpVariable(f"t_{i}", lowBound=0, cat=pulp.LpContinuous) for i in range(N)}

    # --- coverage / validity constraints (EXACT TILING by dyadic squares) ---
    # Root activation
    model += (a_var[root_id] == 1), "root_active"

    # Internal nodes: a_q = s_q + y_q  and  a_child = y_q for each child
    for q in internal_ids:
        model += (a_var[q] == s_var[q] + y_var[q]), f"active_split_or_stop_{q}"
        for c in id2node[q].children:
            model += (a_var[c] == y_var[q]), f"child_activation_{q}_{c}"

    # Leaves: a_l = s_l (cannot split)
    for l in leaf_ids:
        model += (a_var[l] == s_var[l]), f"leaf_active_equals_stop_{l}"
    
    # ----------------------------
    # Enforce GLOBAL stop size bounds:
    # forbid stopping at nodes whose side ∉ [stop_min_side, stop_max_side]
    # (nodes can still be split through)
    # ----------------------------
    for q in all_ids:
        side = id2node[q].size
        if side < stop_min_side or side > stop_max_side:
            model += (s_var[q] == 0), f"forbid_stop_by_global_size_{q}"

    # --- assignment consistency (spec) ---
    # 1) If we stop at q, exactly one block occupies that square.
    for q in all_ids:
        model += (pulp.lpSum(x_var[(i, q)] for i in range(N)) == s_var[q]), f"one_block_per_stop_{q}"

    # 2) Each block is assigned exactly once.
    for i in range(N):
        model += (pulp.lpSum(x_var[(i, q)] for q in all_ids) == 1), f"each_block_once_{i}"

    # (Optional redundant gating: x_{i,q} ≤ s_q)
    for i in range(N):
        for q in all_ids:
            model += (x_var[(i, q)] <= s_var[q]), f"gate_x_by_s_{i}_{q}"

    # --- L1 objective with slacks (spec) ---
    # a_i = ∑_q A(q) x_{i,q}
    targets = total_area * w  # M_eff^2 * w_i
    for i in range(N):
        assigned_area_i = pulp.lpSum(A[q] * x_var[(i, q)] for q in all_ids)
        model += (assigned_area_i - targets[i] <= t_var[i]), f"abs_dev_pos_{i}"
        model += (targets[i] - assigned_area_i <= t_var[i]), f"abs_dev_neg_{i}"

    model += pulp.lpSum(t_var[i] for i in range(N)), "minimize_L1_deviation"

    # --- solver selection ---
    solver_name = (solver_name or "CBC").upper()
    solver: pulp.LpSolver = None
    if solver_name == "CBC":
        solver = pulp.PULP_CBC_CMD(msg=verbose, timeLimit=time_limit)
    elif solver_name == "GUROBI":
        try:
            solver = pulp.GUROBI_CMD(msg=verbose, timeLimit=time_limit)
        except Exception:
            raise RuntimeError("GUROBI solver not available to PuLP.")
    elif solver_name == "GLPK":
        solver = pulp.GLPK_CMD(msg=verbose, options=["--tmlim", str(time_limit or 0)])
    elif solver_name == "SCIP":
        solver = pulp.SCIP_CMD(msg=verbose, timeLimit=time_limit)
    else:
        raise ValueError(f"Unknown solver_name '{solver_name}'. Use 'CBC', 'GUROBI', 'GLPK', or 'SCIP'.")

    # --- solve ---
    status_code = model.solve(solver)
    status_str = pulp.LpStatus[status_code]

    # --- extract solution ---
    # Selected stops
    selected_nodes = [q for q in all_ids if pulp.value(s_var[q]) > 0.5]

    # Pair each selected node with the unique assigned block
    assignments = []
    areas = np.zeros(N, dtype=float)
    for q in selected_nodes:
        # find i with x_{i,q} == 1
        assigned_i = None
        for i in range(N):
            if pulp.value(x_var[(i, q)]) > 0.5:
                assigned_i = i
                break
        if assigned_i is None:
            # should not happen due to constraints
            continue
        nd = id2node[q]
        area_q = A[q]
        areas[assigned_i] = area_q
        assignments.append(
            dict(
                i=assigned_i,
                node_id=q,
                x0=nd.x0,
                y0=nd.y0,
                size=nd.size,
                depth=nd.depth,
                area=area_q,
            )
        )

    abs_errors = np.abs(areas - targets)
    obj_val = float(pulp.value(model.objective))

    return dict(
        status=status_str,
        objective=obj_val,
        M_eff=M_eff,
        L=L,
        areas=areas,
        targets=targets,
        abs_errors=abs_errors,
        assignments=assignments,
        selected_nodes=selected_nodes,
    )


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_quadtree_solution(result, figsize=6):
    M_eff = result["M_eff"]
    assigns = result["assignments"]
    N = len(assigns)

    # Recover normalized weights (these were renormalized in the solver)
    # targets = M_eff^2 * w_i  =>  w_i = targets / M_eff^2
    w_norm = np.array(result["targets"]) / float(M_eff * M_eff)
    w_norm_pct = 100.0 * w_norm / w_norm.sum()

    # assigned area percentage
    assigned_area_pct = 100.0 * result["areas"] / float(M_eff * M_eff)

    # Choose a professional, clear categorical palette (colorblind-friendly)
    cmap = plt.get_cmap("tab10")
    n_colors = getattr(cmap, 'N', 10)
    colors = [cmap(i % n_colors) for i in range(N)]

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.set_xlim(0, M_eff)
    ax.set_ylim(0, M_eff)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Quadtree assignment on {M_eff}x{M_eff} grid")

    for k, a in enumerate(assigns):
        i = a["i"]
        x0, y0, size = a["x0"], a["y0"], a["size"]
        rect = patches.Rectangle(
            (x0, y0),
            size,
            size,
            facecolor=colors[i % n_colors],
            edgecolor="#4c4c4c",
            linewidth=0.8
        )
        ax.add_patch(rect)
        # label with index and normalized information %
        ax.text(
            x0 + size/2,
            y0 + size/2,
            f"{i}: {w_norm_pct[i]:.1f}%\n --> {assigned_area_pct[i]:.1f}%",
            color="black",
            ha="center",
            va="center",
            fontsize=8
        )

    plt.gca().invert_yaxis()  # origin at top-left
    plt.show()


def organize_quadtree_solution(result):
    M_eff = int(result["M_eff"])
    out = {}
    for a in result["assignments"]:
        i = int(a["i"])
        side = int(round((a["area"]) ** 0.5))
        out[i] = (side, int(a["x0"]), int(a["y0"]), M_eff)
    return out