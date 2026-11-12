#!/usr/bin/env python3
"""Graph conditions of Theorem 3 on the test systems.

This file is part of the code that accompanies the paper

    A. Ashrafi, "Identifiability of Grid Admittance Matrices with Zero-Injection
    Buses: Invisible Networks and Conditions for Exact Recovery."

    https://github.com/profashsquared/id_grid_mtx

What it computes
----------------
A group is a maximal set of unsourced buses connected through branches among
them.  Theorem 3 of the paper (Section VI-D) states three graph conditions:

  1. Every unsourced bus has a private neighbor, a sourced neighbor adjacent to
     no other bus of its group.
  2. Every group of two or more buses contains a bus p with a private neighbor i
     such that every other bus of the group has a private neighbor that is not
     adjacent to i.  Every group of one bus has two nonadjacent neighbors.
  3. No two groups share more than one sourced neighbor.

For each test system the script reports the number of unsourced buses, the
group sizes (the row "Largest group" of Table I), and whether the conditions
hold (the row "Graph conditions" of Table I), with the buses that violate them.

For the 14- to 118-bus systems it also checks the idealized case of Theorem 3:
branch weights |y_ij|, one common X/R ratio and no shunts.  Linear programs over
the invisible networks that keep the sign pattern check that the maximum of the
series degree is unique.  Section VII-B reports this for the 30-bus system,
which violates the conditions.

Usage
-----
    python graph_conditions.py data/case14.mat data/case_ieee30.mat data/case57.mat data/case118.mat data/case300.mat \\
        --output graph_conditions.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

import grid_data

IDEALIZED_CHECK_MAX_BUSES = 200   # the idealized checks are skipped for the 300-bus system


def groups(A, u):
    """Groups of unsourced buses: connected components of the unsourced buses."""
    U, seen, out = set(u), set(), []
    for start in u:
        if start in seen:
            continue
        stack, group = [start], []
        seen.add(start)
        while stack:
            x = stack.pop()
            group.append(x)
            for y in np.where(A[x])[0]:
                y = int(y)
                if y in U and y not in seen:
                    seen.add(y)
                    stack.append(y)
        out.append(sorted(group))
    return out


def violations(A, u):
    """List the violations of conditions 1-3; an empty list means that all hold."""
    U, gs, found = set(u), groups(A, u), []
    sourced_neighbors = [set(int(a) for p in g for a in np.where(A[p])[0] if a not in U) for g in gs]
    for i, j in itertools.combinations(range(len(gs)), 2):
        if len(sourced_neighbors[i] & sourced_neighbors[j]) >= 2:
            found.append(("condition 3", gs[i], gs[j]))
    for g in gs:
        members = set(g)

        def private(q):
            return [int(b) for b in np.where(A[q])[0] if b not in U
                    and set(int(x) for x in np.where(A[b])[0] if x in members) == {q}]

        for q in g:
            if not private(q):
                found.append(("condition 1", g, q))
        if len(g) == 1:
            nb = [int(b) for b in np.where(A[g[0]])[0]]
            if not any(not A[x, y] for x, y in itertools.combinations(nb, 2)):
                found.append(("condition 2", g))
        elif not any(all(any(not A[i, k] for k in private(q)) for q in g if q != p)
                     for p in g for i in private(p)):
            found.append(("condition 2", g))
    return found


def cone_rows(L, u):
    """Sign-pattern constraints on the invisible networks Z = L[:, u] X L[u, :].

    The rows are the entries z_ab of Z at pairs (a, b) that are not adjacent, in the
    coordinates of the symmetric matrix X on and above the diagonal.  At such a pair
    the entry of the new network is z_ab alone, and passivity requires z_ab <= 0.
    Also returns tr(Z) as a linear objective.  Without shunts tr(Z) is the change of
    the series degree.
    """
    U = len(u)
    pairs = [(p, q) for p in range(U) for q in range(p, U)]
    ell = L[:, u]
    adjacent = np.abs(L) > 1e-12
    np.fill_diagonal(adjacent, False)
    support = np.where(np.abs(ell).sum(1) > 0)[0]
    rows = []
    for i, a in enumerate(support):
        for b in support[i + 1:]:
            if adjacent[a, b]:
                continue
            r = np.array([ell[a, p] * ell[b, q] + (ell[a, q] * ell[b, p] if p != q else 0.0) for p, q in pairs])
            if np.abs(r).max() > 1e-14:
                rows.append(r / np.abs(r).max())
    gram = ell.T @ ell
    degree = np.array([gram[p, q] * (1.0 if p == q else 2.0) for p, q in pairs])
    return np.array(rows), degree, pairs


def unique_maximum(L, u, tol=1e-9):
    """True if no nonzero direction keeps the sign pattern without lowering the series degree."""
    A, degree, pairs = cone_rows(L, u)
    A_ub = np.vstack([A, -degree[None, :] / np.abs(degree).max()])
    n = len(pairs)
    for k in range(n):
        for sign in (1.0, -1.0):
            c = np.zeros(n)
            c[k] = -sign
            r = linprog(c, A_ub=A_ub, b_ub=np.zeros(len(A_ub)), bounds=[(-1, 1)] * n, method="highs")
            if r.status != 0 or -r.fun > tol:
                return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_files", nargs="+", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = {"systems": {}}
    for path in args.data_files:
        data = grid_data.load_dataset(path)
        Y, bus = data["Y_true"], data["bus_numbers"]
        u = [int(x) for x in grid_data.unsourced_buses(data["I"])]
        A = np.abs(Y) > 1e-10
        np.fill_diagonal(A, False)
        found = violations(A, u)
        to_bus = lambda x: x if isinstance(x, str) else ([int(bus[i]) for i in x] if isinstance(x, list) else int(bus[x]))
        entry = {"U": len(u), "group_sizes": sorted((len(g) for g in groups(A, u)), reverse=True),
                 "conditions_hold": not found,
                 "violations": [[to_bus(x) for x in v] for v in found][:10]}
        if Y.shape[0] <= IDEALIZED_CHECK_MAX_BUSES:
            W = np.abs(Y)
            np.fill_diagonal(W, 0.0)
            L = np.diag(W.sum(1)) - W            # branch weights |y_ij|, no shunts
            entry["idealized_unique_maximum"] = unique_maximum(L, u)
        out["systems"][path.stem] = entry
        print(path.stem, entry)

    args.output.write_text(json.dumps(out, indent=1))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
