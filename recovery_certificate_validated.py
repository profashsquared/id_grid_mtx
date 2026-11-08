#!/usr/bin/env python3
"""Exact verification of the recovery certificates of Proposition 4 (Table I).

This file is part of the code that accompanies the paper

    A. Ashrafi, "Identifiability of Grid Admittance Matrices with Zero-Injection
    Buses: Invisible Networks and Conditions for Exact Recovery."

    https://github.com/profashsquared/Identifiability-of-Grid-Admittance-Matrices

What it proves
--------------
Proposition 4 of the paper certifies that maximizing the series degree J over
the exact family returns the true network K_star.  The certificate is a set of
weights, at least mu > 0, on the sign constraints at the bus pairs without a
branch.  Let gamma be the smallest singular value of these constraints and eta
the residual of the weights.  The certificate holds when the margin
alpha = mu gamma - eta is positive.  Table I lists alpha / tau, where tau is the
largest |J| over invisible networks of unit Frobenius norm.

For the 14-, 30-, 57- and 118-bus systems this script proves lower bounds on
alpha / tau.  A floating-point linear program supplies the weights.  Every other
step is checked without rounding error.

How
---
1. Weights.  A linear program maximizes the floor t of the weights w = x + t,
   x >= 0, subject to A^T w = q / ||q||, where the rows of A are the absent-pair
   constraints and q is the objective.  It is solved by HiGHS at tolerance 1e-10.
   The weights are candidates only.  Nothing below assumes they are accurate.
2. Exact input.  Each float64 entry of the stored Y_star is a dyadic rational
   and is taken as exact.  The systems have no shunt conductance, but rounding
   leaves conductance row sums as low as -7e-15.  Each diagonal conductance is
   therefore set to g_ii = -sum_{j != i} g_ij, a change below 1e-16 |y_ii|
   (Section VII-A of the paper).  This network satisfies the sign constraints of
   Section II exactly, and the certificates refer to it.  H = -K^{-1} Y_us is then
   computed in exact rational arithmetic.  The constraint rows and the objective
   are exact integers over one common denominator.  Rows that vanish identically
   are verified to be exactly zero.
3. Residual eta.  eta is the exact Euclidean norm of q - A^T w, in coordinates d
   with ||d|| = ||Delta||_F.  The computation uses unnormalized coordinates x
   with d = D x, D = 1 or sqrt(2), so that only D^2 enters.
4. Singular value gamma.  The constraint matrix A_x is rounded to float64, with
   entrywise relative error at most 2^-53 and no underflow, so
   ||A_x - A_hat||_F <= 2^-53 / (1 - 2^-53) ||A_hat||_F.  The matrix
   P = A_hat^T A_hat - g0^2 D^2 is formed exactly.  For a floating-point Cholesky
   factor L of P - c I, the remainder E = P - c I - L L^T is exact, and c > 0 with
   ||E||_F < c proves P positive definite.  Then sigma_min(A_hat D^{-1}) > g0 and
   gamma > g0 - ||A_x - A_hat||_F.
5. Margin.  alpha >= mu gamma_lower - eta_upper and alpha / tau are bounded with
   these exact quantities.  The script checks explicitly that c > 0, mu > 0,
   gamma > 0, tau > 0 and alpha > 0, the conditions the argument uses, and stops
   with an error otherwise.

Square roots are bounded with integer square roots.  Every bound in the console
output and in the JSON fields ending in _lower or _upper is a decimal string
rounded in the safe direction: lower bounds down, upper bounds up.  The other
numeric fields are floating-point diagnostics.

The pairs without a branch are read from the true matrix.  The certificate
therefore verifies recovery for a known network.  The selector never receives
these pairs.

Scope
-----
This is a verifier for the four benchmark systems, not a general certificate
tool.  If the multiplier search of step 1 fails, for example because its floor
is unbounded, the script stops and certifies nothing.  That outcome says nothing
about whether recovery fails.  The 300-bus relaxation has no recovery
certificate (Section VII-D).

Usage
-----
    python recovery_certificate_validated.py --output recovery_certificates.json

The results of the paper are in results/recovery_certificate_validated.json.
The run takes about 10 seconds.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from grid_data import load_dataset, unsourced_buses  # noqa: E402

CASES = ("case14", "case_ieee30", "case57", "case118")
RELATIVE_ROW_CUTOFF = 1e-12
G0_FACTOR = 1 - 1e-4          # g0 = G0_FACTOR * floating-point sigma_min
U53 = Fraction(1, 2 ** 53)
SQRT_BITS = 200


# ---------------------------------------------------------------- exact helpers
def sqrt_bounds(x: Fraction) -> tuple[Fraction, Fraction]:
    """Rational lower and upper bounds on sqrt(x), x >= 0."""
    scale = 1 << (2 * SQRT_BITS)
    s = math.isqrt(x.numerator * scale // x.denominator)
    return Fraction(s, 1 << SQRT_BITS), Fraction(s + 1, 1 << SQRT_BITS)


def directed(x: Fraction, up: bool, digits: int = 8) -> str:
    """x to `digits` significant decimal digits, rounded up (up=True) or down."""
    if x == 0:
        return "0"
    a = abs(x)
    e = math.floor(math.log10(float(a)))
    while Fraction(10) ** e > a:
        e -= 1
    while Fraction(10) ** (e + 1) <= a:
        e += 1
    k = e - digits + 1
    s = x / Fraction(10) ** k
    n = math.ceil(s) if up else math.floor(s)
    return str(Decimal(n).scaleb(k))


def cmul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def cdiv(a, b):
    d = b[0] * b[0] + b[1] * b[1]
    return ((a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d)


def exact_H(Yx, s: np.ndarray, u: np.ndarray):
    """H = -K^{-1} Y_us in exact complex rational arithmetic (Gauss-Jordan).

    Yx is a list of lists of (Re, Im) Fraction pairs."""
    U, S = len(u), len(s)
    aug = [[Yx[a][b] for b in u] + [(-Yx[a][b][0], -Yx[a][b][1]) for b in s] for a in u]
    for col in range(U):
        piv = next(r for r in range(col, U) if aug[r][col] != (0, 0))
        aug[col], aug[piv] = aug[piv], aug[col]
        p = aug[col][col]
        aug[col] = [cdiv(x, p) for x in aug[col]]
        for r in range(U):
            f = aug[r][col]
            if r != col and f != (0, 0):
                aug[r] = [(x[0] - y[0], x[1] - y[1])
                          for x, y in zip(aug[r], (cmul(f, z) for z in aug[col]))]
    return [[aug[p][U + c] for c in range(S)] for p in range(U)]


def to_integers(values):
    """Common denominator D and integers n_k with values_k = n_k / D (exact)."""
    D = 1
    for v in values:
        D = D * v.denominator // math.gcd(D, v.denominator)
    return D, [v.numerator * (D // v.denominator) for v in values]


def dyadic_integers(arr: np.ndarray):
    """float64 array -> (object array of ints N, shift e) with arr = N / 2^e exactly."""
    ratios = [float(x).as_integer_ratio() for x in arr.ravel()]
    e = max(d.bit_length() - 1 for _, d in ratios)
    N = np.array([n << (e - (d.bit_length() - 1)) for n, d in ratios], dtype=object)
    return N.reshape(arr.shape), e


# ---------------------------------------------------- floating-point certificate
def float_certificate(Y, s, u):
    """Floating-point multiplier search of step 1; returns the weights."""
    N, U = len(Y), len(u)
    K = Y[np.ix_(u, u)]
    H = -np.linalg.solve(K, Y[np.ix_(u, s)])
    M = np.zeros((N, U), complex)
    M[s] = H.T
    M[u] = -np.eye(U)
    T = []
    for p in range(U):
        for q in range(p, U):
            F = np.outer(M[:, p], M[:, q])
            T.append(F if p == q else (F + F.T) / np.sqrt(2))
    T = np.stack(T, axis=-1)
    T = np.concatenate((T, -1j * T), axis=-1)
    i, j = np.triu_indices(N, 1)
    absent = Y[i, j] == 0
    ai, aj = i[absent], j[absent]
    rows = np.vstack((T[ai, aj].real, -T[ai, aj].imag))
    norms = np.linalg.norm(rows, axis=1)
    keep = norms > RELATIVE_ROW_CUTOFF * norms.max()
    A = rows[keep]
    m = len(A)
    q = -2 * (T[i, j].real - T[i, j].imag).sum(axis=0)
    q_norm = float(np.linalg.norm(q))
    res = linprog(np.r_[np.zeros(m), -1.0],
                  A_eq=np.c_[A.T, A.T @ np.ones(m)], b_eq=q / q_norm,
                  bounds=[(0, None)] * m + [(None, None)], method="highs",
                  options={"primal_feasibility_tolerance": 1e-10,
                           "dual_feasibility_tolerance": 1e-10})
    if not res.success:
        raise RuntimeError(res.message)
    w = res.x[:m] + res.x[-1]
    return ai, aj, keep, w, q_norm, float(np.linalg.svd(A, compute_uv=False)[-1])


# -------------------------------------------------------------- validated check
def validated(case: str) -> dict:
    data = load_dataset(ROOT / "data" / f"{case}.mat")
    Y = data["Y_true"]
    assert np.array_equal(Y, Y.T), "Y_star must be exactly symmetric"
    N = len(Y)
    u = unsourced_buses(data["I"])
    s = np.setdiff1d(np.arange(N), u)
    U = len(u)
    # Exact true network.  The systems have no shunt conductance, but the stored
    # diagonal conductances carry rounding errors of about 1e-16 relative, which make
    # some row sums of G slightly negative.  Each diagonal conductance is therefore set
    # to the exact negative sum of the off-diagonal conductances of its row.  This
    # network satisfies the sign constraints of Section II exactly.
    G, B = Y.real, -Y.imag
    off = ~np.eye(N, dtype=bool)
    assert (G[off] <= 0).all() and (B[off] <= 0).all() and (np.diag(B) >= 0).all()
    Yx = [[(Fraction(float(Y[a, b].real)), Fraction(float(Y[a, b].imag))) for b in range(N)]
          for a in range(N)]
    diag_change = Fraction(0)
    for a in range(N):
        g_aa = -sum(Yx[a][b][0] for b in range(N) if b != a)
        diag_change = max(diag_change, abs(g_aa - Yx[a][a][0]) / Fraction(abs(Y[a, a])))
        Yx[a][a] = (g_aa, Yx[a][a][1])
    ai, aj, keep, w, q_norm, sigma_float = float_certificate(Y, s, u)

    # Exact M with one common denominator: M = Mi / den.
    H = exact_H(Yx, s, u)
    M = [[(Fraction(0), Fraction(0))] * U for _ in range(N)]
    for c, b in enumerate(s):
        for p in range(U):
            M[b][p] = H[p][c]
    for p, a in enumerate(u):
        M[a][p] = (Fraction(-1), Fraction(0))
    den, flat = to_integers([x for row in M for z in row for x in z])
    Mi = [[(flat[2 * (r * U + p)], flat[2 * (r * U + p) + 1]) for p in range(U)]
          for r in range(N)]
    pairs = [(p, q) for p in range(U) for q in range(p, U)]
    r = len(pairs)
    n = 2 * r
    # Unnormalized basis E_pq = e_p e_q^T + e_q e_p^T (p<q): d = Dg x, Dg = 1 or sqrt 2.
    D2 = [1 if p == q else 2 for p, q in pairs] * 2

    def t_entry(i, j, p, q):           # (T'_pq)_ij * den^2, complex integer
        if p == q:
            return cmul(Mi[i][p], Mi[j][p])
        a, b = cmul(Mi[i][p], Mi[j][q]), cmul(Mi[i][q], Mi[j][p])
        return (a[0] + b[0], a[1] + b[1])

    # Exact rows (integers over den^2), same order as the floating-point rows.
    npairs = len(ai)
    exact_rows, zero_rows = [], 0
    float_index = np.cumsum(keep) - 1
    weights = []
    w_min = Fraction(float(w.min()))
    for k in range(2 * npairs):
        half, idx = divmod(k, npairs)
        t = [t_entry(int(ai[idx]), int(aj[idx]), p, q) for p, q in pairs]
        row = ([x[0] for x in t] + [x[1] for x in t]) if half == 0 \
            else ([-x[1] for x in t] + [x[0] for x in t])
        if not any(row):
            zero_rows += 1
            assert not keep[k], "a row dropped as zero must be exactly zero"
            continue
        exact_rows.append(row)
        weights.append(Fraction(float(w[float_index[k]])) if keep[k] else w_min)
    assert zero_rows == int((~keep).sum()), "kept rows must be exactly nonzero"
    m = len(exact_rows)

    # Exact objective q_x (integers over den^2): J(Z) = q_x^T x.
    colsum = [(sum(Mi[a][p][0] for a in range(N)), sum(Mi[a][p][1] for a in range(N)))
              for p in range(U)]
    c = []
    for p, q in pairs:
        dot = (sum(cmul(Mi[a][p], Mi[a][q])[0] for a in range(N)),
               sum(cmul(Mi[a][p], Mi[a][q])[1] for a in range(N)))
        cs = cmul(colsum[p], colsum[q])
        f = 1 if p == q else 2
        c.append((f * (dot[0] - cs[0]), f * (dot[1] - cs[1])))
    qx = [a - b for a, b in c] + [a + b for a, b in c]

    # Check the float objective against the exact one (d-coordinates).
    # eta: exact residual of q_x - A_x^T w' with w' = tau_hat * w.
    tau_hat = Fraction(q_norm)
    wp = [tau_hat * x for x in weights]
    W_den, W = to_integers(wp)                 # wp_k = W_k / W_den
    resid = [qx[l] * W_den - sum(exact_rows[k][l] * W[k] for k in range(m))
             for l in range(n)]                # (q_x - A_x^T w') * den^2 * W_den
    scale2 = Fraction(1, (den * den * W_den) ** 2)
    eta2 = sum(Fraction(R * R, D2[l]) for l, R in enumerate(resid)) * scale2
    tau2 = sum(Fraction(Q * Q, D2[l]) for l, Q in enumerate(qx)) / Fraction(den ** 4)
    mu = min(wp)
    if mu <= 0:
        raise RuntimeError(f"{case}: multiplier floor mu not positive")

    # gamma: round A_x to float64, A_x = rows / den^2.
    den2 = den * den
    A_hat = np.array([[x / den2 for x in row] for row in exact_rows])   # int/int: correctly rounded
    nz = np.abs(A_hat[A_hat != 0])
    assert nz.min() > 1e-300 and np.isfinite(A_hat).all()
    assert all((A_hat[k, l] == 0) == (exact_rows[k][l] == 0)
               for k in range(m) for l in range(n)), "no entry may underflow to zero"
    Ni, eA = dyadic_integers(A_hat)                  # A_hat = Ni / 2^eA
    normF2 = Fraction(int(sum(int(x) * int(x) for x in Ni.ravel())), 1 << (2 * eA))
    delta_up = U53 / (1 - U53) * sqrt_bounds(normF2)[1]  # >= ||A_x - A_hat||_F

    Dinv = 1 / np.sqrt(np.array(D2, float))
    sigma_hat = float(np.linalg.svd(A_hat * Dinv, compute_uv=False)[-1])
    g0 = Fraction(sigma_hat * G0_FACTOR)
    # P = A_hat^T A_hat - g0^2 D2 (exact, dyadic).  Test P > 0 via Cholesky.
    G = Ni.T.dot(Ni)                                  # exact ints, P scale 2^(2 eA)
    g0n, g0d = g0.numerator, g0.denominator           # g0d is a power of two
    eg = g0d.bit_length() - 1
    E2 = max(2 * eA, 2 * eg)
    P = G * (1 << (E2 - 2 * eA))
    for l in range(n):
        P[l, l] -= D2[l] * g0n * g0n * (1 << (E2 - 2 * eg))
    P_float = np.array([[float(Fraction(int(x), 1 << E2)) for x in row] for row in P])
    lam = float(np.linalg.eigvalsh(P_float)[0])
    c_shift = Fraction(lam / 2)
    L = np.linalg.cholesky(P_float - float(c_shift) * np.eye(n))
    Li, eL = dyadic_integers(L)
    LL = Li.dot(Li.T)                                 # scale 2^(2 eL)
    cn, cd = c_shift.numerator, c_shift.denominator
    ec = cd.bit_length() - 1
    E3 = max(E2, 2 * eL, ec)
    R = P * (1 << (E3 - E2)) - LL * (1 << (E3 - 2 * eL))
    for l in range(n):
        R[l, l] -= cn * (1 << (E3 - ec))
    remF2 = Fraction(int(sum(int(x) * int(x) for x in R.ravel())), 1 << (2 * E3))
    pd = c_shift > 0 and remF2 < c_shift * c_shift
    if not pd:
        raise RuntimeError(f"{case}: positive definiteness not proved")
    gamma_lo = g0 - delta_up
    if gamma_lo <= 0:
        raise RuntimeError(f"{case}: gamma not proved positive")

    eta_lo, eta_up = sqrt_bounds(eta2)
    tau_lo, tau_up = sqrt_bounds(tau2)
    if tau_lo <= 0:
        raise RuntimeError(f"{case}: tau not proved positive")
    alpha_lo = mu * gamma_lo - eta_up
    if alpha_lo <= 0:
        raise RuntimeError(f"{case}: margin not proved positive")
    rel_lo = alpha_lo / tau_up
    lo = lambda x: directed(x, up=False)            # noqa: E731
    hi = lambda x: directed(x, up=True, digits=2)   # noqa: E731
    return {
        "diagonal_change_relative_upper": hi(diag_change),
        "N": N, "U": U, "real_coordinates": n, "rows": m, "zero_rows": zero_rows,
        "common_denominator_bits": den.bit_length(),
        "tau_lower": lo(tau_lo), "tau_float": q_norm,
        "eta_over_tau_upper": hi(eta_up / tau_lo),
        "mu_over_tau_lower": lo(mu / tau_up),
        "sigma_min_float": sigma_float, "sigma_hat": sigma_hat,
        "g0": float(g0), "rounding_on_A_upper": hi(delta_up),
        "cholesky_shift": float(c_shift), "cholesky_remainder_F_upper": hi(sqrt_bounds(remF2)[1]),
        "gamma_lower": lo(gamma_lo),
        "alpha_lower": lo(alpha_lo),
        "relative_margin_lower": lo(rel_lo),
        "proved": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "recovery_certificate_validated.json",
                        help="JSON file for the results")
    args = parser.parse_args()
    out = {}
    for case in CASES:
        r = validated(case)
        out[case] = r
        print(f"{case}: diag change <= {r['diagonal_change_relative_upper']} n={r['real_coordinates']} "
              f"rows={r['rows']} zero_rows={r['zero_rows']} den_bits={r['common_denominator_bits']} "
              f"gamma >= {r['gamma_lower']} (sigma float {r['sigma_min_float']:.6g}, "
              f"rounding <= {r['rounding_on_A_upper']}) mu/tau >= {r['mu_over_tau_lower']} "
              f"eta/tau <= {r['eta_over_tau_upper']} alpha/tau >= {r['relative_margin_lower']} "
              f"chol: shift {r['cholesky_shift']:.1e} rem <= {r['cholesky_remainder_F_upper']}",
              flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
