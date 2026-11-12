"""Reader for the test-system data files used by every script in this repository.

This file is part of the code that accompanies the paper

    A. Ashrafi, "Identifiability of Grid Admittance Matrices with Zero-Injection
    Buses: Invisible Networks and Conditions for Exact Recovery."

    https://github.com/profashsquared/id_grid_mtx

Data files
----------
The folder data/ holds one file per test system:

    data/case14.mat        IEEE 14-bus system
    data/case_ieee30.mat   IEEE 30-bus system
    data/case30.mat        case30, the MATPOWER variant of the 30-bus system (Section VII-C)
    data/case57.mat        IEEE 57-bus system
    data/case118.mat       IEEE 118-bus system
    data/case300.mat       IEEE 300-bus system

The system data are those distributed with MATPOWER (Section VII-A of the
paper).  Each file holds M = 800 operating points computed by AC power flow
(Newton-Raphson, in MATPOWER).  The loads vary by 5% around the base case.
The buses are split into 3 zones, and the load changes are correlated with
coefficient 0.3 within a zone.  The generator outputs are adjusted to balance
the load changes.

Each file is a MATLAB v5 file holding one struct named `data`, with only the
fields that the scripts read:

    data.clean.V_pu          N x M bus voltage phasors (per unit), one column per snapshot
    data.clean.I_pu          N x M injected current phasors (per unit)
    data.clean.S_pu          N x M complex power injections (per unit)
    data.ground_truth.Ybus   N x N true admittance matrix
    data.bus_numbers         the N external bus numbers, in the order of the rows
    data.base_case.branch    MATPOWER branch table (used only to find transformer taps)

Any file with the same layout can be used, including a MATLAB v7.3 (HDF5) file.
The README describes how the data were generated.

Conventions
-----------
The admittance matrix is Y = G - jB, as in the paper and in MATPOWER.  An
inductive series branch has negative off-diagonal entries g_ij and b_ij.
A bus is unsourced (zero injection) when its injected current is zero in every
snapshot.  Bus numbers on the command line of every script are the external
bus numbers of the test system, as they appear in the paper.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse

# Columns of the MATPOWER branch table (0-based).
F_BUS, T_BUS, TAP, BR_STATUS = 0, 1, 8, 10


# ---------------------------------------------------------------------------
# MATLAB v7.3 (HDF5) files
# ---------------------------------------------------------------------------
def _dereference(handle, obj):
    """Follow MATLAB object references until a dataset or group is reached."""
    import h5py

    while isinstance(obj, h5py.Dataset) and h5py.check_dtype(ref=obj.dtype) is not None:
        refs = np.asarray(obj).reshape(-1)
        if refs.size != 1:
            raise ValueError("Expected a scalar MATLAB reference field.")
        obj = handle[refs[0]]
    return obj


def _dense(dataset) -> np.ndarray:
    """Read a dense MATLAB array (complex arrays are stored as real/imag pairs)."""
    value = np.asarray(dataset)
    if value.dtype.names and {"real", "imag"}.issubset(value.dtype.names):
        value = value["real"] + 1j * value["imag"]
    if value.ndim > 1:
        # MATLAB stores arrays column by column, so the axes come out reversed.
        value = np.transpose(value, axes=tuple(reversed(range(value.ndim))))
    return np.asarray(value)


def _sparse(group) -> np.ndarray:
    """Read a MATLAB sparse matrix (stored in compressed-column form) as a dense array."""
    ir = np.asarray(group["ir"]).astype(np.int64).reshape(-1)
    jc = np.asarray(group["jc"]).astype(np.int64).reshape(-1)
    raw = np.asarray(group["data"])
    if raw.dtype.names and {"real", "imag"}.issubset(raw.dtype.names):
        raw = raw["real"] + 1j * raw["imag"]
    values = np.asarray(raw).reshape(-1)
    ncols = len(jc) - 1
    nrows_attr = group.attrs.get("MATLAB_sparse", 0)
    nrows = int(np.asarray(nrows_attr).reshape(-1)[0]) if np.size(nrows_attr) else 0
    if nrows <= 0:
        nrows = int(ir.max() + 1) if ir.size else ncols
    return sparse.csc_matrix((values, ir, jc), shape=(nrows, ncols)).toarray()


def _field(handle, *parts: str) -> np.ndarray:
    """Read the numeric field data.<parts[1]>.<parts[2]>... of an HDF5 MAT file."""
    import h5py

    obj = _dereference(handle, handle[parts[0]])
    for part in parts[1:]:
        obj = _dereference(handle, obj)
        if part not in obj:
            raise KeyError(f"The data file has no field {'.'.join(parts)}.")
        obj = obj[part]
    obj = _dereference(handle, obj)
    if isinstance(obj, h5py.Group):
        if {"ir", "jc", "data"}.issubset(obj.keys()):
            return _sparse(obj)
        raise TypeError(f"Expected numeric data but found the group {obj.name}.")
    return _dense(obj)


def _read_hdf5(path: Path) -> dict:
    import h5py

    with h5py.File(path, "r") as handle:
        data = {
            "V": _field(handle, "data", "clean", "V_pu"),
            "I": _field(handle, "data", "clean", "I_pu"),
            "S": _field(handle, "data", "clean", "S_pu"),
            "Y_true": _field(handle, "data", "ground_truth", "Ybus"),
            "bus_numbers": _field(handle, "data", "bus_numbers").reshape(-1),
        }
        try:
            data["branch"] = _field(handle, "data", "base_case", "branch")
        except KeyError:
            data["branch"] = None
    return data


# ---------------------------------------------------------------------------
# Older MAT files (v7 and earlier)
# ---------------------------------------------------------------------------
def _read_legacy(path: Path) -> dict:
    from scipy.io import loadmat

    raw = loadmat(path, simplify_cells=True)["data"]
    ybus = raw["ground_truth"]["Ybus"]
    branch = raw.get("base_case", {}).get("branch") if isinstance(raw.get("base_case"), dict) else None
    return {
        "V": raw["clean"]["V_pu"],
        "I": raw["clean"]["I_pu"],
        "S": raw["clean"]["S_pu"],
        "Y_true": ybus.toarray() if sparse.issparse(ybus) else ybus,
        "bus_numbers": np.asarray(raw["bus_numbers"]).reshape(-1),
        "branch": None if branch is None else np.asarray(branch, dtype=float),
    }


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------
def load_dataset(path) -> dict:
    """Read a data file.

    Returns a dictionary with the complex arrays V, I, S (N x M) and Y_true (N x N),
    the external bus numbers, and the MATPOWER branch table (or None).
    """
    path = Path(path)
    try:
        import h5py

        is_hdf5 = h5py.is_hdf5(path)
    except ImportError:
        is_hdf5 = False
    data = _read_hdf5(path) if is_hdf5 else _read_legacy(path)

    for key in ("V", "I", "S", "Y_true"):
        data[key] = np.asarray(data[key], dtype=np.complex128)
    data["bus_numbers"] = np.asarray(data["bus_numbers"]).astype(int)
    if not (data["V"].shape == data["I"].shape == data["S"].shape):
        raise ValueError("The voltage, current and power arrays have different sizes.")
    n = data["V"].shape[0]
    if data["Y_true"].shape != (n, n):
        raise ValueError("The admittance matrix does not match the number of buses.")
    return data


def unsourced_buses(current: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """Row indices of the unsourced buses: injected current zero in every snapshot."""
    return np.where(np.abs(current).max(axis=1) < tolerance)[0]


def bus_indices(bus_numbers: np.ndarray, buses) -> list[int]:
    """Convert external bus numbers into 0-based row indices."""
    lookup = {int(b): k for k, b in enumerate(bus_numbers)}
    indices = []
    for b in buses:
        if int(b) not in lookup:
            raise SystemExit(f"Bus {b} is not in the data file.")
        indices.append(lookup[int(b)])
    return indices


def tap_transformer_terminals(data: dict) -> list[int]:
    """Row indices of the terminals of in-service transformers with off-nominal taps."""
    branch = data.get("branch")
    if branch is None:
        raise SystemExit("The data file has no MATPOWER branch table (data.base_case.branch).")
    tap = branch[:, TAP]
    off_nominal = (tap != 0) & (np.abs(tap - 1.0) > 1e-12) & (branch[:, BR_STATUS] > 0)
    ends = np.concatenate([branch[off_nominal, F_BUS], branch[off_nominal, T_BUS]])
    return sorted(set(bus_indices(data["bus_numbers"], ends.astype(int))))
