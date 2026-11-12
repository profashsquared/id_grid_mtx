#!/usr/bin/env python3
"""Exact-data maximum of the series degree for large systems (sparse version).

This file is part of the code that accompanies the paper

    A. Ashrafi, "Identifiability of Grid Admittance Matrices with Zero-Injection
    Buses: Invisible Networks and Conditions for Exact Recovery."

    https://github.com/profashsquared/id_grid_mtx

What it computes
----------------
The model is the one of exact_family.py: the family Y(K) of networks that
reproduce exact data, in the real coordinates of the unsourced block K, with the
true network at the origin.  An invisible network changes only the entries among
the unsourced buses and their neighbors, the node set R.  This version therefore
builds the sign constraints sparsely and only on R x R.  Every entry outside
R x R is fixed by the data.  It is used for the 300-bus system (Section VII-D
and Table I of the paper) and for the runs with the row-sum and diagonal
constraints removed (Section VII-B).

The script maximizes the series degree J = J_b + J_g over the passive members,
the program (17) of the paper, and reports the error, the false and
missed branches, and the face width of the optimum in four random directions.
Options:

  --declare-capacitive I J    the branch I-J is a declared series capacitor
                              (Section VI-E): b_IJ >= 0 replaces b_IJ <= 0, b_ii >= 0
                              is dropped at I and J, and the objective counts |b_IJ|
  --release-rowsum-at-taps    drop sum_j g_ij >= 0 at the terminals of the
                              transformers with off-nominal taps
  --release-diag BUS ...      drop b_ii >= 0 at these buses
  --no-rowsum, --no-diag      drop all row-sum or all diagonal constraints
  --exclude-pair I J          the pair I-J is known to carry no branch
  --certificate I J           smallest and largest of w_IJ = -g_IJ - b_IJ over the
                              passive members, with w_IJ = -g_IJ + b_IJ on a declared
                              capacitive pair (Proposition 1, Section VI-E)

A pair outside R x R is fixed by the data.  There an excluded pair must have
y_IJ = 0, and a declared pair must have b_IJ >= 0; otherwise the script stops.
  --dimension STALL           lower bound on the dimension of K_phys from random
                              vertices, stopping after STALL vertices without a
                              rank increase; also counts the branches that vary
  --workers P                 solve the random-vertex programs in P parallel
                              processes (Gurobi only); the results are the same
  --from-data                 form H and Gamma from the measured voltages and
                              currents (not from Y_star), set entries of H below
                              their error scale to zero, relax every sign constraint
                              by a slack set from the measured unsourced currents, and
                              count a branch above 100 times the slack
  --exact-dimension           exact dimension of K_phys from its implicit
                              equalities, found with a few relative-interior programs
  --solver gurobi             use Gurobi (as for the 300-bus system) instead of HiGHS

Bus numbers are the external numbers of the test system.

Usage, with the commands behind the results of the paper
--------------------------------------------------------
Table I, identification error of the maximum from the measurements, 14- to 118-bus systems:
    python exact_family_sparse.py data/case57.mat --solver gurobi --from-data --output case57_max.json
(without --from-data the same command gives the maximum of the exact family, Section VII-B)

Section VII-B, row-sum and diagonal constraints removed (57- and 118-bus systems):
    python exact_family_sparse.py data/case57.mat --no-rowsum --no-diag --dimension 20 --output a.json

Section VII-D and Table I, 300-bus system (Gurobi):
    common="--solver gurobi --release-rowsum-at-taps --declare-capacitive 120 1201"
    python exact_family_sparse.py data/case300.mat $common --exclude-pair 128 168 --output a.json
    python exact_family_sparse.py data/case300.mat $common --exclude-pair 128 168 --from-data --output b.json   (Table I error)
    python exact_family_sparse.py data/case300.mat $common --exclude-pair 128 168 --dimension 10 --workers 5 --output c.json
    python exact_family_sparse.py data/case300.mat $common --exclude-pair 128 168 --exact-dimension --output f.json
    python exact_family_sparse.py data/case300.mat $common --certificate 128 168 --output d.json
    python exact_family_sparse.py data/case300.mat --solver gurobi --release-rowsum-at-taps \\
        --release-diag 1201 --exclude-pair 128 168 --output e.json      (capacitor not declared)

The --dimension run gives a lower bound on the dimension of K_phys from random
vertices.  With --dimension 10, the run stops after 282 vertices with a bound of
214, and 474 branches vary; with --workers 5 it gives the same numbers about
three times faster.  The --exact-dimension run gives the exact dimension, 268,
in about 11 minutes on a 10-core computer.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

import grid_data

HIGHS_OPTIONS = {"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9}
BRANCH_THRESHOLD = 1e-6   # a branch is counted when |y_ij| exceeds this fraction of the largest one
FACE_TOLERANCE = 1e-9     # relative slack that defines the optimal face of the maximum
SOLVER = "highs"


class LPResult:
    def __init__(self, status, x=None, fun=None, message=""):
        self.status, self.x, self.fun, self.message = status, x, fun, message


GUROBI_PARAMS = {"OutputFlag": 0, "FeasibilityTol": 1e-9, "OptimalityTol": 1e-9, "Method": 2, "Crossover": 1}


def _solve_gurobi(c, A, b):
    import gurobipy as gp

    env = gp.Env(params=GUROBI_PARAMS | {"Threads": 4})
    m = gp.Model(env=env)
    x = m.addMVar(len(c), lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
    m.addConstr(sp.csr_matrix(A) @ x <= b)
    m.setObjective(np.asarray(c) @ x, gp.GRB.MINIMIZE)
    m.optimize()
    return _gurobi_result(gp, m, x)


def _gurobi_result(gp, m, x):
    if m.Status == gp.GRB.OPTIMAL:
        return LPResult(0, x.X.copy(), m.ObjVal, "optimal")
    if m.Status == gp.GRB.INFEASIBLE:
        return LPResult(2, message="infeasible")
    if m.Status == gp.GRB.UNBOUNDED:
        return LPResult(3, message="unbounded")
    if m.Status == gp.GRB.INF_OR_UNBD:
        m.setParam("DualReductions", 0)
        m.optimize()
        return LPResult(2 if m.Status == gp.GRB.INFEASIBLE else 3, message=f"Gurobi status {m.Status}")
    return LPResult(4, message=f"Gurobi status {m.Status}")


def solve(c, A, b):
    """Minimize c x subject to A x <= b, with free variables."""
    if SOLVER == "gurobi":
        return _solve_gurobi(c, A, b)
    bounds = [(None, None)] * len(c)
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs", options=HIGHS_OPTIONS)
    if res.status not in (0, 2, 3):
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs-ds")
    return res


# ---------------------------------------------------------------------------
# Many linear programs with the same constraints, in parallel (Gurobi only)
# ---------------------------------------------------------------------------
# Measured on the 300-bus system (Apple M1 Max, 10 cores): one barrier solve takes
# about 0.9 s with 4 threads.  Reusing one model and changing only the objective saves
# almost nothing, and a warm-started simplex is slower for random objectives (6 s
# primal, 67 s dual).  Five processes with 2 threads each solve about 3.2 programs per
# second, against 1.0 for one process.
_WORKER = None


def _worker_init(A, b, threads):
    global _WORKER
    import gurobipy as gp

    m = gp.Model(env=gp.Env(params=GUROBI_PARAMS | {"Threads": threads}))
    x = m.addMVar(A.shape[1], lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
    m.addConstr(A @ x <= b)
    _WORKER = (gp, m, x)


def _worker_solve(c):
    gp, m, x = _WORKER
    m.setObjective(np.asarray(c) @ x, gp.GRB.MINIMIZE)
    m.reset()
    m.optimize()
    r = _gurobi_result(gp, m, x)
    return r.status, r.x, r.fun


class LPPool:
    """Solves min c x subject to A x <= b for a list of objectives c.

    With workers > 1 and Gurobi, the programs run in parallel processes, each with
    its own copy of the model; otherwise they run one after another."""

    def __init__(self, A, b, workers=1):
        self.A, self.b, self.workers = A, b, workers
        self.pool = None
        if workers > 1 and SOLVER == "gurobi":
            import multiprocessing
            import os
            threads = max(1, (os.cpu_count() or 2) // workers)
            self.pool = multiprocessing.get_context("spawn").Pool(
                workers, initializer=_worker_init, initargs=(A, b, threads))

    def map(self, objectives):
        if self.pool is None:
            return [solve(c, self.A, self.b) for c in objectives]
        results = [LPResult(s, x, f) for s, x, f in self.pool.map(_worker_solve, objectives)]
        # A rare solve with fewer threads ends in numerical trouble (for example a false
        # "unbounded").  Such a program is solved again exactly as without workers.
        return [r if r.status == 0 else solve(c, self.A, self.b) for r, c in zip(results, objectives)]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


def _solve_with_bounds(c, A, b, lb, ub, extra=None):
    """Minimize c z subject to A z <= b and lb <= z <= ub (infinite entries are free).

    extra holds additional Gurobi parameters for this solve."""
    if SOLVER == "gurobi":
        import gurobipy as gp

        inf = gp.GRB.INFINITY
        m = gp.Model(env=gp.Env(params=GUROBI_PARAMS | {"Threads": 8} | (extra or {})))
        z = m.addMVar(len(c), lb=np.where(np.isfinite(lb), lb, -inf), ub=np.where(np.isfinite(ub), ub, inf))
        m.addConstr(A @ z <= b)
        m.setObjective(np.asarray(c) @ z, gp.GRB.MINIMIZE)
        m.optimize()
        return _gurobi_result(gp, m, z)
    bounds = [(None if not np.isfinite(lo) else lo, None if not np.isfinite(hi) else hi) for lo, hi in zip(lb, ub)]
    return linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs", options=HIGHS_OPTIONS)


# Settings tried in turn for each relative-interior program.  Barrier without crossover
# is fastest and gives a point inside the optimal face, which makes more slacks strictly
# positive in each round; the point is accepted only if it satisfies the constraints.
# Otherwise crossover with careful numerics gives an exact vertex.  (On the full program,
# with the rows that are strict at the true network, crossover ran for more than 45
# minutes on the 300-bus system; on the cone used here it takes 3 to 40 s per round.)
RELATIVE_INTERIOR_ATTEMPTS = (
    {"Crossover": 0},
    {"Crossover": 1, "NumericFocus": 2},
)


def exact_dimension(A, b, tol=1e-7, verbose=False):
    """Exact dimension of the polyhedron P = {x : A x <= b}, which contains x = 0.

    Every constraint with b_i > 0 is strict at x = 0, so near x = 0 the polyhedron
    coincides with the cone {x : A_0 x <= 0} of the constraints with b_i = 0, which is
    the cone of Theorem 3.  A convex set that contains x = 0 has the same affine hull
    as its intersection with any neighborhood of x = 0, so P and this cone have the
    same dimension.  Only the rows with b_i = 0 are used, and x is boxed to [-1, 1],
    which leaves the affine hull unchanged and keeps the programs bounded.

    A constraint is an implicit equality when it holds with equality at every point.
    Each round solves one linear program over the remaining candidates C,
        max sum_{i in C} t_i  subject to  A_0 x + t <= 0 on C,  A_0 x <= 0 elsewhere,  0 <= t <= 1,
    and removes every i whose slack at the point found exceeds tol: it is strict
    somewhere.  When the maximum is zero, the remaining candidates are the implicit
    equalities, and the dimension is the number of coordinates minus their rank.  A few
    rounds replace one program per constraint.  The rows of A are scaled to a largest
    coefficient of 1.
    """
    A = sp.csr_matrix(A)[np.where(np.asarray(b) <= 0)[0]].copy()
    # Coefficients below 1e-9 are rounding left over from cancellation, for example in
    # the row sums; they lie below the feasibility tolerance and make the programs badly
    # conditioned, so they are set to zero.
    A.data[np.abs(A.data) < 1e-9] = 0.0
    A.eliminate_zeros()
    m, n = A.shape
    zero = np.zeros(m)
    C = np.arange(m)
    rounds, smallest_removed, largest_kept = 0, np.inf, 0.0
    while len(C):
        rounds += 1
        S = sp.csr_matrix((np.ones(len(C)), (C, np.arange(len(C)))), shape=(m, len(C)))
        cost = np.r_[np.zeros(n), -np.ones(len(C))]
        lb = np.r_[np.full(n, -1.0), np.zeros(len(C))]
        ub = np.r_[np.full(n, 1.0), np.ones(len(C))]
        AS = sp.hstack([A, S]).tocsr()
        report = []
        for extra in (RELATIVE_INTERIOR_ATTEMPTS if SOLVER == "gurobi" else (None,)):
            started = time.time()
            r = _solve_with_bounds(cost, AS, zero, lb, ub, extra)
            if verbose:
                print(f"   round {rounds}, |C| = {len(C)}, settings {extra}: {getattr(r, 'message', r.status)} "
                      f"in {time.time() - started:.1f} s", flush=True)
            if r.status != 0:
                report.append(f"status {getattr(r, 'message', r.status)}")
                continue
            # Strictness is read from the actual slacks at the point found, which must be feasible.
            slack = -(A @ r.x[:n])
            violation = float(max(0.0, -slack.min()))
            if verbose:
                print(f"      largest violation {violation:.1e}", flush=True)
            if violation <= 1e-8:          # an order of magnitude below the strictness tolerance
                break
            report.append(f"violation {violation:.1e}")
        else:
            raise RuntimeError("relative-interior program failed: " + "; ".join(report))
        tC = slack[C]
        strict = tC > tol
        if not strict.any():
            largest_kept = float(tC.max()) if len(tC) else 0.0
            break
        smallest_removed = min(smallest_removed, float(tC[strict].min()))
        C = C[~strict]
    implicit = np.unique(np.round(A[C].toarray(), 12), axis=0)
    sv = np.linalg.svd(implicit, compute_uv=False) if len(implicit) else np.zeros(0)
    rank = int((sv > 1e-8 * sv[0]).sum()) if len(sv) else 0
    return {"dimension": n - rank, "implicit_equalities": int(len(C)), "rank": rank, "rounds": rounds,
            "smallest_slack_of_removed": smallest_removed, "largest_slack_of_implicit": largest_kept,
            "singular_values_at_cut": (sv[max(0, rank - 2):rank + 2] / sv[0]).tolist() if len(sv) else []}


class SparseFamily:
    """Sign constraints and objectives of the family, restricted to the node set R.

    Indices are 0-based rows of Y.  For every coordinate of K the change of the
    entries of G and B on R x R is stored in the sparse matrix D, whose rows are,
    in order: the off-diagonal g_ij of R, the off-diagonal b_ij of R, the
    conductance row sums of R, and the diagonal b_ii of R.
    """

    def __init__(self, Y, s, u, declared=(), rowsum=True, diag=True,
                 release_rowsum=(), release_diag=(), exclude=(), H=None, slack=0.0):
        # Y is the member at the origin of the coordinates: the true network, or with
        # --from-data the zero-padded Kron reduction formed from the data.  H is formed
        # from Y unless it is given (from the data).  slack relaxes every sign constraint.
        N, U = Y.shape[0], len(u)
        if H is None:
            H = -np.linalg.solve(Y[np.ix_(u, u)], Y[np.ix_(u, s)])
        M = np.zeros((N, U), complex)
        M[s, :] = H.T
        M[u, :] = -np.eye(U)
        R = np.where(np.abs(M).max(axis=1) > 1e-13)[0]      # unsourced buses and their neighbors
        self.N, self.U, self.R, self.Y = N, U, R, Y
        MR, nR = M[R], len(R)
        pairs = [(p, q) for p in range(U) for q in range(p, U)]
        half = len(pairs)
        self.nv = 2 * half
        iu = np.triu_indices(nR, 1)
        self.iu_R = iu
        npair = len(iu[0])
        self.npair = npair

        columns = []
        for k, (p, q) in enumerate(pairs):
            mp, mq = MR[:, p], MR[:, q]
            T = np.outer(mp, mp) if p == q else np.outer(mp, mq) + np.outer(mq, mp)
            for offset, dY in ((0, T), (half, -1j * T)):        # K changes by +a, or by -j d
                dG, dB = dY.real, -dY.imag
                v = np.concatenate([dG[iu], dB[iu], dG.sum(axis=1), dB.diagonal()])
                nz = np.where(np.abs(v) > 1e-14)[0]
                columns.append((offset + k, nz, v[nz]))
        rows = np.concatenate([nz for _, nz, _ in columns])
        cols = np.concatenate([np.full(len(nz), j) for j, nz, _ in columns])
        vals = np.concatenate([v for _, _, v in columns])
        D = sp.csr_matrix((vals, (rows, cols)), shape=(2 * npair + 2 * nR, self.nv))
        self.D = D
        Dg, Db = D[:npair], D[npair:2 * npair]
        Drowsum, Ddiag = D[2 * npair:2 * npair + nR], D[2 * npair + nR:]

        G0, B0 = Y.real, -Y.imag
        GR, BR = G0[np.ix_(R, R)], B0[np.ix_(R, R)]
        position = {int(r): k for k, r in enumerate(R)}

        def pair_index(a, b):
            if a not in position or b not in position:
                return None
            lo, hi = sorted((position[a], position[b]))
            return int(np.where((iu[0] == lo) & (iu[1] == hi))[0][0])

        # sign = -1 on declared capacitive pairs: b_ij >= 0 there, and |b_ij| in the objective.
        # Declared and excluded pairs outside R x R are fixed by the data and add no
        # constraint; main() checks that the data agree with them.
        self.sign = np.ones(npair)
        for a, b in declared:
            k = pair_index(a, b)
            if k is not None:
                self.sign[k] = -1.0

        blocks = [(Dg, -GR[iu]), (sp.diags(self.sign) @ Db, -self.sign * BR[iu])]   # g_ij <= 0, b_ij <= 0
        if rowsum:                                                                   # sum_j g_ij >= 0
            keep = np.array([r not in set(release_rowsum) for r in R])
            blocks.append((-Drowsum[keep], G0.sum(axis=1)[R][keep]))
        if diag:                                                                     # b_ii >= 0
            keep = np.array([r not in set(release_diag) for r in R])
            blocks.append((-Ddiag[keep], BR.diagonal()[keep]))
        for a, b in exclude:                                                         # y_ab = 0
            k = pair_index(a, b)
            if k is not None:
                for Dpart, v0 in ((Dg, GR[iu][k]), (Db, BR[iu][k])):
                    blocks.append((Dpart[k], np.array([-v0])))
                    blocks.append((-Dpart[k], np.array([v0])))
        A = sp.vstack([a for a, _ in blocks]).tocsr()
        b = np.concatenate([v for _, v in blocks])
        if slack:
            # Data-driven slack: each sign constraint may be violated by the error of the
            # data.  A row sum adds up to |R| entries, so its slack is scaled by sqrt(|R|).
            scale = np.ones(len(b))
            if rowsum:
                n_rowsum = len(blocks[2][1])
                scale[2 * npair:2 * npair + n_rowsum] = np.sqrt(nR)
            b = b + slack * scale
        norm = np.asarray(abs(A).max(axis=1).todense()).ravel()
        keep = norm > 1e-13
        self.truth_violation = float(max(0.0, -b[keep].min()))   # > 0 if the true network is not passive here
        A = sp.diags(1.0 / norm[keep]) @ A[keep]
        b = b[keep] / norm[keep]
        b[np.abs(b) < 1e-11 * max(1.0, np.abs(b).max())] = 0.0
        self.A, self.b = A.tocsr(), b
        self.pair_index = pair_index

        # Objectives on R (pairs outside R x R are fixed and add a constant).
        self.c_b = np.asarray((-2 * sp.diags(self.sign) @ Db).sum(axis=0)).ravel()   # J_b, or J_|b|
        self.c0_b = float(-2 * (self.sign * BR[iu]).sum())
        self.c_g = np.asarray((-2 * Dg).sum(axis=0)).ravel()                         # J_g
        self.c0_g = float(-2 * GR[iu].sum())

    def Y_of(self, x):
        """Full admittance matrix of the member with coordinates x."""
        npair, nR = self.npair, len(self.R)
        dG = np.asarray(self.D[:npair] @ x).ravel()
        dB = np.asarray(self.D[npair:2 * npair] @ x).ravel()
        dG_rowsum = np.asarray(self.D[2 * npair:2 * npair + nR] @ x).ravel()
        dB_diag = np.asarray(self.D[2 * npair + nR:] @ x).ravel()
        GR, BR = np.zeros((nR, nR)), np.zeros((nR, nR))
        GR[self.iu_R] = dG
        GR += GR.T
        BR[self.iu_R] = dB
        BR += BR.T
        np.fill_diagonal(GR, dG_rowsum - GR.sum(axis=1))
        np.fill_diagonal(BR, dB_diag)
        Y = self.Y.copy()
        Y[np.ix_(self.R, self.R)] += GR - 1j * BR
        return Y


def relative_error(Y_hat, Y):
    return float(np.linalg.norm(Y_hat - Y) / np.linalg.norm(Y))


def main():
    global SOLVER
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_file", type=Path)
    ap.add_argument("--declare-capacitive", nargs=2, type=int, action="append", default=[], metavar=("I", "J"))
    ap.add_argument("--release-rowsum-at-taps", action="store_true")
    ap.add_argument("--release-diag", nargs="*", type=int, default=[], metavar="BUS")
    ap.add_argument("--no-rowsum", action="store_true")
    ap.add_argument("--no-diag", action="store_true")
    ap.add_argument("--exclude-pair", nargs=2, type=int, action="append", default=[], metavar=("I", "J"))
    ap.add_argument("--certificate", nargs=2, type=int, action="append", default=[], metavar=("I", "J"))
    ap.add_argument("--dimension", type=int, default=0, metavar="STALL")
    ap.add_argument("--solver", choices=("highs", "gurobi"), default="highs")
    ap.add_argument("--exact-dimension", action="store_true")
    ap.add_argument("--workers", type=int, default=1, metavar="P")
    ap.add_argument("--from-data", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    SOLVER = args.solver

    data = grid_data.load_dataset(args.data_file)
    Y, bus = data["Y_true"], data["bus_numbers"]
    N = Y.shape[0]
    u = grid_data.unsourced_buses(data["I"])
    s = np.setdiff1d(np.arange(N), u)
    index = lambda pairs: [tuple(grid_data.bus_indices(bus, p)) for p in pairs]
    declared = index(args.declare_capacitive)
    release_rowsum = grid_data.tap_transformer_terminals(data) if args.release_rowsum_at_taps else []
    release_diag = set(grid_data.bus_indices(bus, args.release_diag)) | {i for p in declared for i in p}
    # With --from-data, H and Gamma are formed from the measured voltages and currents
    # alone, and Y_star is used only to score the result.
    H_data, slack, Ybase, from_data = None, 0.0, Y, {}
    if args.from_data:
        if args.exact_dimension:
            raise SystemExit("--exact-dimension needs the true network at the origin; run it without --from-data")
        V, Q = data["V"], data["I"]
        P = np.linalg.pinv(V[s])
        H_data, Gam = V[u] @ P, Q[s] @ P
        Gam = (Gam + Gam.T) / 2          # nearest symmetric matrix: the model is reciprocal
        sig = np.linalg.svd(V[s], compute_uv=False)
        # Error scale of H: the part of V_U outside the row space of V_S, divided by the
        # smallest singular value of V_S.  Entries of H below it are set to zero.
        tau_H = float(np.abs(V[u] - H_data @ V[s]).max() / sig[-1])
        H_data[np.abs(H_data) <= tau_H] = 0.0
        # Slack: the largest measured current at an unsourced bus (zero in the model) over
        # the smallest singular value of V_S.
        slack = float(np.abs(Q[u]).max() / sig[-1])
        Ybase = np.zeros((N, N), complex)
        Ybase[np.ix_(s, s)] = Gam
        from_data = {"sigma_min_VS": float(sig[-1]), "error_scale_H": tau_H, "slack": slack,
                     "branch_count_threshold": 100 * slack}
        print("from data:", from_data)
    fam = SparseFamily(Ybase, s, u, declared=declared, rowsum=not args.no_rowsum, diag=not args.no_diag,
                       release_rowsum=release_rowsum, release_diag=release_diag,
                       exclude=index(args.exclude_pair), H=H_data, slack=slack)
    # A pair outside R x R is fixed by the data, so an exclusion or a declaration there
    # adds no constraint.  It must then agree with the fixed entry, within the slack.
    tol = slack + 1e-12 * np.abs(fam.Y).max()
    for (a, b), (i, j) in zip(args.exclude_pair, index(args.exclude_pair)):
        y = fam.Y[i, j]
        if fam.pair_index(i, j) is None and max(abs(y.real), abs(y.imag)) > tol:
            raise SystemExit(f"--exclude-pair {a} {b}: the data fix y = {y:.6g}, which is not zero")
    for (a, b), (i, j) in zip(args.declare_capacitive, declared):
        b_fixed = -fam.Y[i, j].imag
        if fam.pair_index(i, j) is None and b_fixed < -tol:
            raise SystemExit(f"--declare-capacitive {a} {b}: the data fix b = {b_fixed:.6g}, which is negative")
    print(f"N={N} U={len(u)} |R|={len(fam.R)} coordinates={fam.nv} constraints={fam.A.shape[0]}")
    out = {"data_file": str(args.data_file), "N": N, "U": len(u), "unsourced_buses": bus[u].tolist(),
           "R_size": int(len(fam.R)), "coordinates_of_K": fam.nv, "options": vars(args) | {"output": str(args.output),
           "data_file": str(args.data_file)}, "rowsum_released_at": bus[release_rowsum].tolist(),
           "diag_released_at": bus[sorted(release_diag)].tolist(), "truth_violation": fam.truth_violation,
           "from_data": from_data}

    # Branch certificates for the requested pairs.
    if args.certificate:
        out["certificates"] = []
        for a, b in args.certificate:
            i, j = grid_data.bus_indices(bus, (a, b))
            k = fam.pair_index(i, j)
            if k is None:
                out["certificates"].append({"pair": [a, b], "status": "fixed by the data"})
                continue
            # w = -g_ij - b_ij, or -g_ij + b_ij on a declared capacitive pair (sign = -1).
            # With g = Re{y} and b = -Im{y}, w = -Re{y} + sign Im{y}.
            sign = fam.sign[k]
            w = lambda y: float(-y.real + sign * y.imag)
            cw = -np.asarray((fam.D[k] + sign * fam.D[fam.npair + k]).todense()).ravel()   # change of w
            w0 = w(fam.Y[i, j])      # value at the origin of the coordinates
            lo, hi = solve(cw, fam.A, fam.b), solve(-cw, fam.A, fam.b)
            bound = lambda r, s: float(w0 + s * r.fun) if r.status == 0 else (
                "unbounded" if r.status == 3 else f"status {r.status}")
            rec = {"pair": [a, b], "true_value": w(Y[i, j]), "min": bound(lo, 1), "max": bound(hi, -1)}
            print("certificate:", rec)
            out["certificates"].append(rec)

    # Maximum of the series degree.
    c = fam.c_b + fam.c_g
    c0 = fam.c0_b + fam.c0_g
    res = solve(-c, fam.A, fam.b)
    entry = {"status": int(res.status), "message": str(res.message)}
    if res.status == 0:
        Y_hat = fam.Y_of(res.x)
        value = float(c @ res.x)
        scale = np.abs(c).max()
        A2 = sp.vstack([fam.A, sp.csr_matrix(-c / scale)]).tocsr()
        b2 = np.concatenate([fam.b, [(-value + FACE_TOLERANCE * max(1.0, abs(value))) / scale]])
        rng = np.random.default_rng(0)
        width = 0.0
        for _ in range(4):
            e = rng.standard_normal(fam.nv)
            lo, hi = solve(e, A2, b2), solve(-e, A2, b2)
            width = float("inf") if lo.status or hi.status else max(width, -hi.fun - lo.fun)
        iu = np.triu_indices(N, 1)
        truth = np.abs(Y[iu]) > 1e-10
        count = 100 * slack if args.from_data else BRANCH_THRESHOLD * np.abs(Y_hat[iu]).max()
        est = np.abs(Y_hat[iu]) > count
        pair = lambda k: [int(bus[iu[0][k]]), int(bus[iu[1][k]])]
        entry.update({"optimum": c0 + value, "true_value": c0, "relative_error": relative_error(Y_hat, Y),
                      "face_width": width, "true_pairs": int(truth.sum()),
                      "false_branches": [pair(k) for k in np.where(est & ~truth)[0]],
                      "missed_branches": [pair(k) for k in np.where(truth & ~est)[0]]})
        print(f"max J: error {entry['relative_error']:.2e}, face width {width:.1e}, "
              f"false {len(entry['false_branches'])}, missed {len(entry['missed_branches'])}, "
              f"true bus pairs {entry['true_pairs']}")
    else:
        print(f"max J: {res.message}")
    out["maximum"] = entry

    # Lower bound on the dimension of K_phys from random vertices.
    if args.dimension:
        rng = np.random.default_rng(1)
        X, rank, stall, n_lp = [], 0, 0, 0
        with LPPool(fam.A, fam.b, args.workers) as pool:
            while stall < args.dimension:
                # The objectives are drawn in the same order as one at a time; results
                # past the stopping point of a batch are discarded.
                batch = [rng.standard_normal(fam.nv) for _ in range(max(1, args.workers))]
                for r in pool.map(batch):
                    n_lp += 1
                    if r.status == 3:
                        rank, stall = float("inf"), args.dimension
                        break
                    if r.status != 0:
                        continue
                    X.append(r.x)
                    new = 0
                    if len(X) > 1:
                        sv = np.linalg.svd(np.array(X[1:]) - X[0], compute_uv=False)
                        new = int((sv > 1e-7 * max(1.0, sv[0])).sum())
                    stall = stall + 1 if new == rank else 0
                    rank = new
                    if stall >= args.dimension:
                        break
        varying = 0
        if len(X) > 1 and rank != float("inf"):
            spread = np.array([np.abs(fam.D[:2 * fam.npair] @ (x - X[0])).ravel() for x in X[1:]]).max(axis=0)
            varying = int(((spread[:fam.npair] > 1e-7) | (spread[fam.npair:] > 1e-7)).sum())
        out["K_phys"] = {"dimension_lower_bound": rank, "vertices": len(X), "linear_programs": n_lp,
                         "branches_that_vary": varying, "candidate_pairs_in_R": fam.npair,
                         "candidate_pairs": N * (N - 1) // 2}
        print("K_phys:", out["K_phys"])

    if args.exact_dimension:
        out["K_phys_exact"] = exact_dimension(fam.A, fam.b)
        print("K_phys, exact:", out["K_phys_exact"])

    args.output.write_text(json.dumps(out, indent=1))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
