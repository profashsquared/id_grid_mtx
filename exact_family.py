#!/usr/bin/env python3
"""Exact-data analysis of the networks that fit the data (14-, 30-, 57- and 118-bus systems).

This file is part of the code that accompanies the paper

    A. Ashrafi, "Identifiability of Grid Admittance Matrices with Zero-Injection
    Buses: Invisible Networks and Conditions for Exact Recovery."

    https://github.com/profashsquared/Identifiability-of-Grid-Admittance-Matrices

What it computes
----------------
With exact data, the symmetric admittance matrices that reproduce the measured
voltages and currents form the family Y(K) of the paper.  K is the block of Y
among the unsourced buses.  A member differs from the true network Y_star by an
invisible network, which changes only the entries at the unsourced buses and at
their neighbors.  The family is described here by the real and imaginary parts
of the entries of K on and above the diagonal, U(U+1) real coordinates for U
unsourced buses.  The coordinates are measured from K_star, so the origin is the
true network.  The matrix H of the paper is formed from Y_star.

The passive members are those with inductive series branches, nonnegative shunt
conductances and b_ii >= 0, the sign constraints of Section II.  They form the
polyhedron K_phys.  The script reports, for one test system (Section VII-B):

  * the size of the family and the dimension of K_phys (Table I);
  * the certificate of every candidate branch, that is, the smallest and largest
    value of its series conductance plus susceptance over K_phys, and the number
    of ambiguous branches, whose presence the data and passivity do not decide
    (Table I);
  * the maximum of the series degree J = J_b + J_g over K_phys, the program
    (17) of the paper, with its error (Section VII-B) and its face width, the
    uniqueness check of Section IV.

Every program is a linear program, solved with HiGHS through SciPy at
feasibility tolerance 1e-10.

The same script gives the results of Section VII-C for the case30 variant.
There the maximum of J does not exist, and the certificate of branch 9-11 has
no upper bound.

Usage
-----
    python exact_family.py DATA.mat --output RESULT.json

Examples (one data file per test system):
    python exact_family.py data/case14.mat --output case14_exact.json
    python exact_family.py data/case30.mat --output case30_exact.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

import grid_data

LP_OPTIONS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}
BRANCH_THRESHOLD = 1e-6   # a branch is counted when |y_ij| exceeds this fraction of the largest one
FACE_TOLERANCE = 1e-9     # relative slack that defines the optimal face of a maximum
STATUS = {0: "optimal", 2: "infeasible", 3: "unbounded"}


def solve(c, A, b):
    """Minimize c x subject to A x <= b, with free variables (HiGHS)."""
    bounds = [(None, None)] * len(c)
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs", options=LP_OPTIONS)
    if res.status not in (0, 2, 3):
        # Numerical trouble at the tight tolerances: retry with the dual simplex.
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs-ds")
    return res


class Family:
    """The family Y(x) = Y0 + sum_k x_k T_k in the real coordinates x of K.

    Y0 is the origin of the coordinates (the true network).  For coordinate k,
    which belongs to the entry (p, q) of K, the matrix T_k is M E_pq M^T with
    E_pq the symmetric unit matrix at (p, q).  M has the rows H^T at the sourced
    buses and -I at the unsourced buses, so that M (K - K_star) M^T is the
    invisible network that moves the unsourced block from K_star to K.
    The first half of x changes Re(K) (conductance), the second half Im(K).
    """

    def __init__(self, Y0: np.ndarray, H: np.ndarray, s: np.ndarray, u: np.ndarray):
        N, U = Y0.shape[0], len(u)
        self.N, self.U = N, U
        M = np.zeros((N, U), complex)
        M[s, :] = H.T
        M[u, :] = -np.eye(U)
        self.pairs = [(p, q) for p in range(U) for q in range(p, U)]
        half = len(self.pairs)
        self.nv = 2 * half
        self.G0, self.B0 = Y0.real.copy(), -Y0.imag.copy()
        self.dG = np.zeros((N, N, self.nv))
        self.dB = np.zeros((N, N, self.nv))
        for k, (p, q) in enumerate(self.pairs):
            E = np.zeros((U, U))
            E[p, q] = E[q, p] = 1.0
            T = M @ E @ M.T
            # K changes by +a at (p, q):    Y changes by a T
            self.dG[:, :, k] = T.real
            self.dB[:, :, k] = -T.imag
            # K changes by -j d at (p, q):  Y changes by -j d T
            self.dG[:, :, half + k] = T.imag
            self.dB[:, :, half + k] = T.real
        self.iu = np.triu_indices(N, 1)
        self._sign_constraints()

    def _sign_constraints(self):
        """Passivity as A x <= b: g_ij <= 0, b_ij <= 0, sum_j g_ij >= 0, b_ii >= 0."""
        iu, di = self.iu, np.arange(self.N)
        rows = [self.dG[iu], self.dB[iu], -self.dG.sum(axis=1), -self.dB[di, di]]
        rhs = [-self.G0[iu], -self.B0[iu], self.G0.sum(axis=1), np.diag(self.B0)]
        A, b = np.vstack(rows), np.concatenate(rhs)
        norm = np.abs(A).max(axis=1)
        keep = norm > 1e-13          # rows with zero coefficients are entries fixed by the data
        self.A = A[keep] / norm[keep, None]
        self.b = b[keep] / norm[keep]
        # Right-hand sides that are zero up to roundoff (true entries that sit exactly
        # on a sign constraint) are set to exactly zero.
        roundoff = 1e-11 * max(1.0, float(np.abs(self.b).max()))
        self.b[np.abs(self.b) < roundoff] = 0.0

    def matrices(self, x):
        return self.G0 + self.dG @ x, self.B0 + self.dB @ x

    def Y(self, x):
        G, B = self.matrices(x)
        return G - 1j * B

    def objective(self):
        """Series degree J = J_b + J_g as (coefficients, constant): value = coefficients @ x + constant."""
        iu = self.iu
        J_b = (-2 * self.dB[iu].sum(axis=0), -2 * self.B0[iu].sum())   # 2 x sum of series susceptances
        J_g = (-2 * self.dG[iu].sum(axis=0), -2 * self.G0[iu].sum())   # 2 x sum of series conductances
        return J_b[0] + J_g[0], J_b[1] + J_g[1]


def branch_errors(Y_hat, Y_true, bus):
    """Relative error, and false and missed branches (as bus-number pairs)."""
    iu = np.triu_indices(Y_true.shape[0], 1)
    truth = np.abs(Y_true[iu]) > 1e-10 * max(1.0, np.abs(Y_true[iu]).max())
    mag = np.abs(Y_hat[iu])
    est = mag > BRANCH_THRESHOLD * max(mag.max(), np.finfo(float).eps)
    pair = lambda k: [int(bus[iu[0][k]]), int(bus[iu[1][k]])]
    return {
        "relative_error": float(np.linalg.norm(Y_hat - Y_true) / np.linalg.norm(Y_true)),
        "false_branches": [pair(k) for k in np.where(est & ~truth)[0]],
        "missed_branches": [pair(k) for k in np.where(truth & ~est)[0]],
    }


def face_width(fam, c, x_opt):
    """Largest range of any coordinate of K over the optimal face of max c x."""
    value = float(c @ x_opt)
    scale = float(np.abs(c).max())
    A = np.vstack([fam.A, -c / scale])
    b = np.concatenate([fam.b, [(-value + FACE_TOLERANCE * max(1.0, abs(value))) / scale]])
    width = 0.0
    for k in range(fam.nv):
        e = np.zeros(fam.nv)
        e[k] = 1.0
        lo, hi = solve(e, A, b), solve(-e, A, b)
        if lo.status != 0 or hi.status != 0:
            return float("inf")
        width = max(width, -hi.fun - lo.fun)
    return float(width)


def dimension_of_K_phys(fam):
    """Exact affine dimension of K_phys.

    K_phys is first checked for boundedness, coordinate by coordinate.  A sign
    constraint a x <= b is an implicit equality when it holds with equality at
    every member of K_phys, that is, when the minimum of a x over K_phys is b.
    Only constraints that are tight at the true network (b = 0) can be implicit
    equalities.  The dimension of K_phys is the number of coordinates minus the
    rank of the implicit equalities.

    The extreme points along the coordinates alone are not enough: for the
    118-bus system they span one direction, while K_phys has dimension 2.
    """
    for k in range(fam.nv):
        for sign in (1.0, -1.0):
            e = np.zeros(fam.nv)
            e[k] = sign
            if solve(e, fam.A, fam.b).status != 0:
                return {"bounded": False}
    A, b = fam.A, fam.b
    _, first = np.unique(np.round(np.hstack([A, b[:, None]]), 12), axis=0, return_index=True)
    implicit = []
    for i in np.sort(first):                  # each distinct constraint once
        if b[i] > 1e-9:
            continue                          # strict at the true network
        r = solve(A[i], A, b)                 # minimum of a_i x over K_phys
        if r.status == 0 and r.fun >= b[i] - 1e-9:
            implicit.append(i)
    rank = 0
    if implicit:
        sv = np.linalg.svd(A[implicit], compute_uv=False)
        rank = int((sv > 1e-8 * sv[0]).sum())
    return {"bounded": True, "dimension": fam.nv - rank, "implicit_equalities": len(implicit),
            "rank_of_implicit_equalities": rank}


def certificates(fam, Y_true, bus):
    """Range of w_ij = -(g_ij + b_ij) over K_phys for every candidate branch (i, j)."""
    iu = fam.iu
    w0 = -(fam.G0[iu] + fam.B0[iu])
    dw = -(fam.dG[iu] + fam.dB[iu])
    tol = 1e-7 * np.abs(Y_true[iu]).max()
    truth = np.abs(Y_true[iu]) > 1e-10 * np.abs(Y_true[iu]).max()
    counts = {"candidates": len(w0), "always_present": 0, "always_absent": 0, "ambiguous": 0,
              "ambiguous_true_branches": 0}
    ambiguous = []
    for e in range(len(w0)):
        if not np.any(dw[e]):
            lo = hi = w0[e]                       # fixed by the data
        else:
            a, c = solve(dw[e], fam.A, fam.b), solve(-dw[e], fam.A, fam.b)
            lo = w0[e] + a.fun if a.status == 0 else -np.inf
            hi = w0[e] - c.fun if c.status == 0 else np.inf
        if lo > tol:
            counts["always_present"] += 1
        elif hi <= tol:
            counts["always_absent"] += 1
        else:
            counts["ambiguous"] += 1
            counts["ambiguous_true_branches"] += int(truth[e])
            ambiguous.append({"branch": [int(bus[iu[0][e]]), int(bus[iu[1][e]])],
                              "true_value": float(max(w0[e], 0.0)),
                              "min": float(max(lo, 0.0)), "max": float(hi) if np.isfinite(hi) else "unbounded"})
    counts["ambiguous_branches"] = ambiguous
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_file", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    data = grid_data.load_dataset(args.data_file)
    Y_true, bus = data["Y_true"], data["bus_numbers"]
    N = Y_true.shape[0]
    u = grid_data.unsourced_buses(data["I"])
    s = np.setdiff1d(np.arange(N), u)
    U = len(u)
    print(f"N={N}, unsourced U={U}: buses {bus[u].tolist()}")

    K_true = Y_true[np.ix_(u, u)]
    H = -np.linalg.solve(K_true, Y_true[np.ix_(u, s)])
    fam = Family(Y_true, H, s, u)

    results = {"data_file": str(args.data_file), "N": N, "U": U,
               "unsourced_buses": bus[u].tolist(), "coordinates_of_K": fam.nv}

    # Maximum of the series degree.
    c, c0 = fam.objective()
    res = solve(-c, fam.A, fam.b)
    entry = {"status": STATUS.get(res.status, str(res.status))}
    if res.status == 0:
        entry["optimum"] = float(c0 + c @ res.x)
        entry["face_width"] = face_width(fam, c, res.x)
        entry.update(branch_errors(fam.Y(res.x), Y_true, bus))
        print(f"max J: error {entry['relative_error']:.2e}, face width {entry['face_width']:.1e}, "
              f"false {len(entry['false_branches'])}, missed {len(entry['missed_branches'])}")
    else:
        print(f"max J: {entry['status']}")
    results["maximum"] = entry

    results["K_phys"] = dimension_of_K_phys(fam)
    print("K_phys:", results["K_phys"])

    results["certificates"] = certificates(fam, Y_true, bus)
    print("certificates:", {k: v for k, v in results["certificates"].items() if k != "ambiguous_branches"})
    for entry in results["certificates"]["ambiguous_branches"]:
        print("   ambiguous branch", entry)

    args.output.write_text(json.dumps(results, indent=1))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
