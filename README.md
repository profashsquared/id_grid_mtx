# Identifiability of Grid Admittance Matrices with Zero-Injection Buses

This repository holds the code and data for the following paper:

> A. Ashrafi, "Identifiability of Grid Admittance Matrices with Zero-Injection Buses: Invisible Networks and Conditions for Exact Recovery."

## Requirements

- Python 3 (tested with 3.12), with NumPy and SciPy. h5py is needed only to read MATLAB v7.3 files.
- The linear programs use HiGHS through SciPy.
- Gurobi (`gurobipy`) is needed for two things:
  - the noisy-data study;
  - any run with `--solver gurobi`, such as the 300-bus runs.

## Data

The folder `data/` holds one file per test system. The systems are the IEEE 14-, 30-, 57-, 118- and 300-bus systems, plus `case30`, MATPOWER's variant of the 30-bus system.

Each file holds 800 AC power-flow operating points, along with the true admittance matrix. Each is a MATLAB v5 file that holds only the fields the scripts read. `grid_data.py` reads the files and describes their layout.

The data were generated as follows.
- The system data are those distributed with MATPOWER.
- Each operating point is an AC power flow, solved by Newton–Raphson.
- The loads are scaled by random factors with a standard deviation of 5% around the base case.
- The load buses are split into 3 zones. The load factors are correlated with coefficient 0.3 within a zone. The zones are the MATPOWER areas for `case30` and a spectral partition of the network for the other systems.
- The generators share each load change in proportion to their reserve in the direction of the change.
- Operating points that violate a generator's real-power limits are discarded. Reactive-power and voltage limits are not enforced.
- The stored voltages and currents carry no added noise. The noisy-data study adds its own noise to the injected powers.

## Scripts

| Script | What it computes | In the paper |
|---|---|---|
| `grid_data.py` | Reader for the data files | — |
| `exact_family.py` | Exact family of one system, the dimension of K_phys and the branch certificates | Table I, Sections VII-B and VII-C |
| `exact_family_sparse.py` | Maximum of the series degree, from the exact family or from the measurements; the 300-bus runs | Table I, Sections VII-B and VII-D |
| `graph_conditions.py` | Graph conditions of Theorem 3, and uniqueness of the maximum in the idealized case | Table I, Section VII-B |
| `recovery_certificate_validated.py` | Exact verification of the recovery margins of Proposition 4 | Table I, Section VII-B |
| `noisy_identification.py` | Identification from noisy injections at 30, 40 and 50 dB | Table II, Section VII-E |

The docstring of each script gives the exact command behind each result.

## Recovery margins of Table I

To reproduce the margins, run:

```bash
python recovery_certificate_validated.py --output recovery_certificates.json
```

The run takes about 10 seconds. The output should match `results/recovery_certificate_validated.json`.

The certified lower bounds on α/τ are as follows:

| System | Lower bound on α/τ | Table I |
|---|---:|---:|
| 14-bus | 0.52203193 | 0.52 |
| 30-bus | 0.017111987 | 0.017 |
| 57-bus | 0.0071239474 | 0.0071 |
| 118-bus | 0.042442370 | 0.042 |

### How the margins are proved

Proposition 4 gives the certificate: weights, all at least μ > 0, on the sign constraints at the bus pairs without a branch. γ is the smallest singular value of these constraints, and η is the residual of the weights. The maximum of the series degree J is then the true network when the margin α = μγ − η is positive. τ is the largest |J| over invisible networks of unit Frobenius norm.

1. **Weights.** A floating-point linear program finds the weights. They are only candidates, and no later step assumes they are accurate.
2. **Exact input.** Each stored float64 entry is a dyadic rational, so it is taken as exact.
   - Rounding leaves some conductance row sums as low as −7·10⁻¹⁵.
   - Each diagonal conductance is therefore set to g_ii = −Σ_{j≠i} g_ij. This changes it by less than 10⁻¹⁶|y_ii|.
   - The resulting network satisfies the sign constraints exactly, and the certificates refer to it (Section VII-A).
   - H, the constraint rows and the objective are computed in exact rational arithmetic.
3. **Residual.** η is computed exactly.
4. **Singular value.** γ is bounded from below rigorously.
   - The constraint matrix is rounded to float64, and the rounding error is bounded.
   - P = ÂᵀÂ − g₀²D² is formed exactly.
   - A floating-point Cholesky factor of P − cI leaves an exact remainder E. The conditions c > 0 and ‖E‖_F < c prove that P is positive definite.
5. **Margin.** The script checks that c, μ, γ, τ and α are all positive. If any check fails, it stops with an error.

Square roots are bounded with integer square roots. Every bound that is printed, or stored in a JSON field ending in `_lower` or `_upper`, is rounded in the safe direction: lower bounds down, upper bounds up.

**Scope.** The script verifies only the four benchmark systems. It is not a general certificate tool.
- If the multiplier search fails, the script certifies nothing. That outcome says nothing about whether recovery fails.
- The 300-bus relaxation has no recovery certificate.
