#!/usr/bin/env python3
"""Maximum series-degree identification from noisy injections (Table II).

This file is part of the code that accompanies the paper

    A. Ashrafi, "Identifiability of Grid Admittance Matrices with Zero-Injection
    Buses: Invisible Networks and Conditions for Exact Recovery."

    https://github.com/profashsquared/id_grid_mtx

Copyright (c) 2026 Ashkan Ashrafi.  Released under the MIT License.  See the
file LICENSE in the root of the repository.

Measurement model (Section VII-E of the paper)
----------------------------------------------
The voltages are exact.  Every entry of the complex power injections S receives
circular complex Gaussian noise of variance sigma^2, with
SNR = 10 log10( mean |S_im|^2 / sigma^2 ).  The measured currents are
I_meas = conj(S_meas / V), entry by entry.  The zero injections of the unsourced
buses are perturbed like every other entry.

The estimator knows which buses are unsourced, as in state estimation.  It does
not impose their zero injections as constraints.  It uses the unsourced buses
for two things only: to add candidate branches around them (step 1 below), and
to set the noise level from their measured currents, which contain only noise.

Estimator
---------
The unknown is the whole symmetric admittance matrix Y.  The equation Y V = I
is replaced by a residual band.  The residual Y V - I_meas is projected onto the
right singular vectors of V whose singular values exceed the noise level.  The
noise level is 10 sqrt(N) times the RMS measured current at the unsourced buses,
divided by the median magnitude of the self-admittances of the sourced buses
found in step 1.

  1. Candidate branches.  A row-by-row pass applies the rule of the author's
     patent to each bus: a small linear program finds the smallest projected
     residual of that row under its sign constraints, and a second one
     maximizes the row's series degree within (1 + eps) of it.  A branch becomes
     a candidate if its magnitude exceeds 0.2% of the geometric mean of the
     largest branch magnitudes at its two buses.  The row of an unsourced bus
     has no scale, so its self-susceptance is set to the median of those of the
     sourced buses.  Every pair that joins an unsourced bus to another unsourced
     bus, or to a neighbor of the unsourced buses found by this pass, is added,
     because the rows of the unsourced buses cannot locate those branches.
  2. Joint program on the candidates.  A first linear program finds the
     smallest worst-case projected residual r* over the passive networks.  A
     second maximizes the series degree J over the passive networks whose
     residual is within (1 + eps) r*, with eps = 1%.
  3. Support and refit.  A branch (i, j) is kept if |y_ij| exceeds 5% of the
     geometric mean of the largest branch magnitudes at buses i and j.  The kept
     branches are refitted by minimum residual.

The oracle is the same minimum-residual fit on the true branch set.  After the
refit a branch is counted when its magnitude exceeds 1% of the largest one.
The linear programs are solved with Gurobi.

Solver threads
--------------
A thread is one stream of computation on one processor core.  Gurobi can split
a linear program over several threads that run at the same time (option
--threads).  The number of threads should not change the answer, but here it
can, for two reasons.

  * Rounding.  Splitting the work changes the order of the arithmetic.  With 4
    threads the computer may add numbers as (a + b) + (c + d) instead of
    ((a + b) + c) + d.  In floating point these differ in the last digits,
    around 1e-16.
  * Ties.  The maximum of J within the residual band is not always a single
    point.  Several networks can reach the same maximum to within the solver
    tolerance.  The tiny rounding differences then decide which of these
    equally good points Gurobi returns.

Every returned point solves the linear program.  After it, the support rule and
the refit can keep a slightly different set of branches, so the error of one
realization can change.  For trial 0 of the 14-bus system at 40 dB, the refit
error is 10.1% with 2 threads and 6.7% with 4.  The means over the realizations
change little.  For the 14-bus system at 40 dB the mean refit error is 11.8%,
11.6% and 11.0% with 1, 2 and 4 threads.

Table II was computed with 2 threads, the default here.  With it, Table II is
reproduced exactly.  The 118-bus system takes more than an hour.  The times in
Table II depend on the computer.  The exact-family maxima of Section VII-B are
unique, as the recovery certificates of Table I prove, so they do not depend on
the number of threads.

Usage
-----
Table II and the 30 dB results of Section VII-E (10 realizations, 5 for the
118-bus system), likewise for data/case_ieee30.mat and data/case57.mat:
    python noisy_identification.py data/case14.mat --snr 30 40 50 --trials 10 --oracle --output case14_noise.json
    python noisy_identification.py data/case118.mat --snr 30 40 50 --trials 5 --oracle --output case118_noise.json
The output holds the means of Table II and every realization, from which the
worst realizations and the medians of Section VII-E follow.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB

import grid_data

COUNT_THRESHOLD = 0.01   # after the refit, a branch is counted above 1% of the largest one


# ---------------------------------------------------------------------------
# Noise, support rule and error measures
# ---------------------------------------------------------------------------
def add_noise(S, V, snr_db, rng):
    """Noisy power injections, returned as measured currents conj(S_meas / V)."""
    sigma2 = np.mean(np.abs(S) ** 2) / 10 ** (snr_db / 10)
    eta = np.sqrt(sigma2 / 2) * (rng.standard_normal(S.shape) + 1j * rng.standard_normal(S.shape))
    return np.conj((S + eta) / V)


def branch_support(Y, threshold):
    """Keep (i, j) if |y_ij| exceeds `threshold` times the geometric mean of the
    largest off-diagonal magnitudes in rows i and j."""
    mag = np.abs(Y - np.diag(np.diag(Y)))
    rowmax = np.maximum(mag.max(axis=1), 1e-300)
    return mag > threshold * np.sqrt(np.outer(rowmax, rowmax))


def metrics(Y_hat, Y_true):
    """Relative error, false and missed branches, and F1 score."""
    iu = np.triu_indices(Y_true.shape[0], 1)
    truth = np.abs(Y_true[iu]) > 1e-10
    mag = np.abs(Y_hat[iu])
    est = mag > COUNT_THRESHOLD * mag.max()
    tp, fp, fn = int((truth & est).sum()), int((~truth & est).sum()), int((truth & ~est).sum())
    return {"relative_error": float(np.linalg.norm(Y_hat - Y_true) / np.linalg.norm(Y_true)),
            "FP": fp, "FN": fn, "F1": 2 * tp / max(2 * tp + fp + fn, 1)}


def gurobi_env(threads):
    return gp.Env(params={"OutputFlag": 0, "Threads": threads, "NumericFocus": 1, "DualReductions": 0,
                          "Method": 2, "Crossover": 1})


# ---------------------------------------------------------------------------
# Step 1: row-by-row pass that proposes the candidate branches
# ---------------------------------------------------------------------------
class RowPass:
    """Row-wise linear programs on the projected data (rank cut at 1e-10)."""

    def __init__(self, V, I_meas, eps):
        Uv, sig, Wh = np.linalg.svd(V, full_matrices=False)
        r = int((sig > 1e-10 * sig[0]).sum())
        self.A = Uv[:, :r] * sig[:r]               # projected voltages: row residual = y_m A - B_m
        self.B = I_meas @ Wh[:r].conj().T
        self.N, self.eps = V.shape[0], eps
        self.env = gurobi_env(threads=1)

    def _row_lp(self, m, objective, band=None, self_susceptance=None):
        """Row m with x = [Re y_m, Im y_m].  objective "band": smallest residual;
        "degree": largest series degree of the row within the given band."""
        N = self.N
        mdl = gp.Model(env=self.env)
        lb, ub = np.full(2 * N, -GRB.INFINITY), np.full(2 * N, GRB.INFINITY)
        off = np.ones(N, bool)
        off[m] = False
        ub[:N][off] = 0.0           # g_mj <= 0
        lb[N:][off] = 0.0           # Im y_mj >= 0, that is b_mj <= 0
        ub[N + m] = 0.0             # Im y_mm <= 0, that is b_mm >= 0
        if self_susceptance is not None:
            # The row of an unsourced bus solves y_m V = 0, which has no scale: fix b_mm.
            lb[N + m] = ub[N + m] = -self_susceptance
        x = mdl.addMVar(2 * N, lb=lb, ub=ub)
        rowsum = np.zeros(2 * N)
        rowsum[:N] = 1.0
        mdl.addConstr(rowsum @ x >= 0)                                   # shunt conductance >= 0
        Ar, Ai = self.A.real, self.A.imag
        P = np.block([[Ar.T, -Ai.T], [Ai.T, Ar.T]])                      # real form of y_m A
        res = P @ x - np.concatenate([self.B[m].real, self.B[m].imag])
        if objective == "band":
            t = mdl.addVar(lb=0.0)
            mdl.addConstr(res <= t)
            mdl.addConstr(-res <= t)
            mdl.setObjective(t, GRB.MINIMIZE)
        else:
            mdl.addConstr(res <= band)
            mdl.addConstr(-res <= band)
            c = np.zeros(2 * N)
            c[:N][off] = -1.0
            c[N:][off] = 1.0                                             # sum_j (-g_mj - b_mj)
            mdl.setObjective(c @ x, GRB.MAXIMIZE)
        mdl.optimize()
        if mdl.Status != GRB.OPTIMAL:
            return None, None
        xv = x.X
        return xv[:N] + 1j * xv[N:], (mdl.ObjVal if objective == "band" else None)

    def rows(self, buses, self_susceptance=None):
        """Solve the given rows (smallest residual, then largest degree); other rows stay zero."""
        Y = np.zeros((self.N, self.N), complex)
        for m in buses:
            y, tau = self._row_lp(m, "band", self_susceptance=self_susceptance)
            if y is None:
                continue
            band = (1 + self.eps) * tau + 1e-9 * float(np.abs(self.B).max())
            y2, _ = self._row_lp(m, "degree", band=band, self_susceptance=self_susceptance)
            Y[m] = y if y2 is None else y2
        return Y


def symmetrize(Y, solved):
    """Average y_ij and y_ji where both rows were solved; copy where only one was."""
    N = Y.shape[0]
    out = np.zeros_like(Y)
    for i in range(N):
        for j in range(N):
            if solved[i] and solved[j]:
                out[i, j] = 0.5 * (Y[i, j] + Y[j, i])
            elif solved[i]:
                out[i, j] = Y[i, j]
            elif solved[j]:
                out[i, j] = Y[j, i]
    return out


# ---------------------------------------------------------------------------
# Steps 2 and 3: joint program on the candidate branches, support and refit
# ---------------------------------------------------------------------------
class BandProgram:
    """Joint program in the symmetric Y, restricted to the candidate branches.

    With the thin SVD V = U Sig W^H, truncated at rank_tol, the projected residual
    is R = Y U Sig - I_meas W.  The band is on the real and imaginary parts of R.
    """

    def __init__(self, V, I_meas, candidate, rank_tol, threads):
        Uv, sig, Wh = np.linalg.svd(V, full_matrices=False)
        r = int((sig > rank_tol * sig[0]).sum())
        self.rank = r
        self.A = Uv[:, :r] * sig[:r]
        self.B = I_meas @ Wh[:r].conj().T
        self.N, self.candidate, self.threads = V.shape[0], candidate, threads
        self._linear_maps()

    def _linear_maps(self):
        """Sparse maps from x = [Re y; Im y] (candidate entries and diagonal) to residuals and row sums."""
        N, A, cand = self.N, self.A, self.candidate
        r = A.shape[1]
        entries = [(a, b) for a in range(N) for b in range(a, N) if a == b or cand[a, b] or cand[b, a]]
        pos = {ab: k for k, ab in enumerate(entries)}
        n = len(entries)
        self.entries, self.n = entries, n
        neighbors = [[k for k in range(N) if (min(i, k), max(i, k)) in pos] for i in range(N)]
        rows, cols, vals = [], [], []
        row = 0
        for i in range(N):
            for j in range(r):
                for k in neighbors[i]:
                    ar, ai = A[k, j].real, A[k, j].imag
                    kk = pos[(min(i, k), max(i, k))]
                    rows += [row, row, row + 1, row + 1]
                    cols += [kk, n + kk, kk, n + kk]
                    vals += [ar, -ai, ai, ar]
                row += 2
        self.P = sp.csr_matrix((vals, (rows, cols)), shape=(2 * N * r, 2 * n))
        self.rhs = np.stack([self.B.real, self.B.imag], axis=-1).reshape(-1)
        rows, cols = [], []
        for a in range(N):
            for b in neighbors[a]:
                rows.append(a)
                cols.append(pos[(min(a, b), max(a, b))])
        self.rowsum = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, 2 * n))
        self.off = np.array([a != b for a, b in entries])
        c = np.zeros(2 * n)
        c[:n][self.off] = -2.0
        c[n:][self.off] = 2.0                                  # J = 2 sum_(i<j) (-g_ij - b_ij)
        self.cJ = c

    def _model(self, objective, band=None, support=None):
        n = self.n
        m = gp.Model(env=gurobi_env(self.threads))
        lb, ub = np.full(2 * n, -GRB.INFINITY), np.full(2 * n, GRB.INFINITY)
        ub[:n][self.off] = 0.0            # g_ij <= 0
        lb[n:][self.off] = 0.0            # b_ij <= 0
        ub[n:][~self.off] = 0.0           # b_ii >= 0
        if support is not None:
            for k, (a, b) in enumerate(self.entries):
                if a != b and not support[a, b]:
                    lb[k] = ub[k] = lb[n + k] = ub[n + k] = 0.0
        x = m.addMVar(2 * n, lb=lb, ub=ub)
        m.addConstr(self.rowsum @ x >= 0)                     # shunt conductance >= 0
        res = self.P @ x - self.rhs
        if objective == "residual":
            t = m.addVar(lb=0.0)
            m.addConstr(res <= t)
            m.addConstr(-res <= t)
            m.setObjective(t, GRB.MINIMIZE)
        else:
            m.addConstr(res <= band)
            m.addConstr(-res <= band)
            m.setObjective(self.cJ @ x, GRB.MAXIMIZE)
        m._x = x
        return m

    def _Y(self, m):
        xv, n = m._x.X, self.n
        Y = np.zeros((self.N, self.N), complex)
        for k, (a, b) in enumerate(self.entries):
            Y[a, b] = Y[b, a] = xv[k] + 1j * xv[n + k]
        return Y

    def refit(self, Y, threshold):
        """Keep the branches of Y selected by the support rule and refit by minimum residual."""
        m = self._model("residual", support=branch_support(Y, threshold))
        m.optimize()
        return self._Y(m) if m.Status == GRB.OPTIMAL else None

    def maximum(self, eps):
        """Smallest residual r*, then the maximum of J within (1 + eps) r*."""
        m1 = self._model("residual")
        m1.optimize()
        if m1.Status != GRB.OPTIMAL:
            return None, None
        r_star = float(m1.ObjVal)
        m2 = self._model("degree", band=(1 + eps) * r_star + 1e-9 * float(np.abs(self.B).max()))
        m2.optimize()
        if m2.Status != GRB.OPTIMAL:
            return None, r_star
        return self._Y(m2), r_star


# ---------------------------------------------------------------------------
# One realization
# ---------------------------------------------------------------------------
def identify(V, I_meas, u, Y_true, eps=0.01, candidate_threshold=0.002, refit_threshold=0.05,
             oracle=False, threads=2):
    N = V.shape[0]
    s = np.setdiff1d(np.arange(N), u)
    out = {}

    # Step 1: candidate branches.
    rp = RowPass(V, I_meas, eps)
    Y_rows = rp.rows(s)
    solved = np.zeros(N, bool)
    solved[s] = True
    Y_cand = symmetrize(Y_rows, solved)
    scale = float(np.median(-np.diag(Y_rows)[s].imag))       # typical self-susceptance of a sourced bus
    Y_unsourced = rp.rows(u, self_susceptance=scale)
    Y_cand[u] = Y_unsourced[u]
    Y_cand = np.where(np.abs(Y_cand) > np.abs(Y_cand).T, Y_cand, Y_cand.T)
    cand = branch_support(Y_cand, candidate_threshold)
    cand = cand | cand.T
    R = np.union1d(u, np.where(cand[u].any(axis=0))[0])      # unsourced buses and their neighbors
    for p in u:
        cand[p, R] = True
        cand[R, p] = True
    np.fill_diagonal(cand, False)
    truth = np.abs(Y_true) > 1e-10
    out["candidate_branches"] = int(np.triu(cand, 1).sum())
    out["true_branches_not_candidates"] = int((np.triu(truth, 1) & ~np.triu(cand, 1)).sum())

    # Rank cut: keep the directions of V excited above the noise.  The noise level is
    # the RMS measured current at the unsourced buses, which is pure noise.
    sig = np.linalg.svd(V, compute_uv=False)
    sigma_I = float(np.sqrt(np.mean(np.abs(I_meas[u]) ** 2))) if len(u) else 0.0
    y_typical = float(np.median(np.abs(np.diag(Y_rows)[s])))
    rank_tol = max(float(10 * sigma_I * np.sqrt(N) / max(y_typical, 1e-12) / sig[0]), 1e-10)
    out["rank_tolerance"] = rank_tol

    # Steps 2 and 3.
    prog = BandProgram(V, I_meas, cand, rank_tol, threads)
    out["rank"] = int(prog.rank)
    Y_max, r_star = prog.maximum(eps)
    if Y_max is None:
        out["status"] = "failed"
        return out
    out["status"] = "ok"
    out["maximum"] = metrics(Y_max, Y_true) | {"r_star": r_star}
    Y_refit = prog.refit(Y_max, refit_threshold)
    if Y_refit is not None:
        out["refit"] = metrics(Y_refit, Y_true)
    if oracle:
        po = BandProgram(V, I_meas, cand | truth, rank_tol, threads)
        Y_oracle = po.refit(Y_true, 1e-3)
        if Y_oracle is not None:
            out["oracle"] = metrics(Y_oracle, Y_true)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_file", type=Path)
    ap.add_argument("--snr", nargs="+", type=float, default=[40, 50], help="SNR levels in dB")
    ap.add_argument("--trials", type=int, default=10, help="noise realizations per SNR level")
    ap.add_argument("--eps", type=float, default=0.01, help="relative width of the residual band")
    ap.add_argument("--refit", type=float, default=0.05, help="threshold of the support rule")
    ap.add_argument("--candidate", type=float, default=0.002, help="threshold for the candidate branches")
    ap.add_argument("--oracle", action="store_true", help="also fit the true branch set")
    ap.add_argument("--threads", type=int, default=2, help="Gurobi threads for the joint program")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    data = grid_data.load_dataset(args.data_file)
    V, S, Y_true = data["V"], data["S"], data["Y_true"]
    u = grid_data.unsourced_buses(data["I"])
    results = {"data_file": str(args.data_file), "N": int(V.shape[0]), "U": int(len(u)),
               "options": {k: v for k, v in vars(args).items() if k not in ("data_file", "output")}, "runs": []}
    for snr in args.snr:
        for trial in range(args.trials):
            rng = np.random.default_rng(1000 * int(snr) + trial)
            I_meas = add_noise(S, V, snr, rng)
            t0 = time.time()
            r = identify(V, I_meas, u, Y_true, eps=args.eps, candidate_threshold=args.candidate,
                         refit_threshold=args.refit, oracle=args.oracle, threads=args.threads)
            r.update({"snr_db": snr, "trial": trial, "time_s": round(time.time() - t0, 2)})
            results["runs"].append(r)
            line = f"SNR {snr:g} dB, trial {trial}: {r['status']}"
            if "refit" in r:
                f = r["refit"]
                line += f", refit error {100 * f['relative_error']:.1f}%, FP {f['FP']}, FN {f['FN']}, F1 {f['F1']:.2f}"
            if "oracle" in r:
                line += f", oracle {100 * r['oracle']['relative_error']:.1f}%"
            print(line + f", {r['time_s']} s")

    # Means over the realizations, as in Table II.
    results["means"] = {}
    for snr in args.snr:
        runs = [r for r in results["runs"] if r["snr_db"] == snr and "refit" in r]
        if not runs:
            continue
        mean = lambda key, part="refit": float(np.mean([r[part][key] for r in runs if part in r]))
        row = {"refit_error": mean("relative_error"), "FP": mean("FP"), "FN": mean("FN"), "F1": mean("F1"),
               "time_s": float(np.mean([r["time_s"] for r in runs])),
               "exact_branch_sets": int(sum(r["refit"]["FP"] == 0 and r["refit"]["FN"] == 0 for r in runs)),
               "realizations": len(runs)}
        if args.oracle:
            row["oracle_error"] = mean("relative_error", "oracle")
        results["means"][f"{snr:g} dB"] = row
        print(f"mean at {snr:g} dB:", {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})

    args.output.write_text(json.dumps(results, indent=1))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
