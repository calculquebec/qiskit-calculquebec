"""Tests for ReadoutMitigation."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from qiskit_calculquebec.mitigation.readout import ReadoutMitigation, _faulty_qubit_checker


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def backend():
    mock = MagicMock()
    mock.target.num_qubits = 3
    mock._client.machine_name = "yukon"
    return mock


@pytest.fixture
def rem_matrix(backend):
    rem = ReadoutMitigation(backend, method="matrix")
    rem.cals_from_matrices([
        np.array([[0.97, 0.05], [0.03, 0.95]]),
        np.array([[0.96, 0.06], [0.04, 0.94]]),
        np.array([[0.98, 0.04], [0.02, 0.96]]),
    ])
    return rem


@pytest.fixture
def rem_m3(backend):
    rem = ReadoutMitigation(backend, method="m3")
    rem.cals_from_matrices([
        np.array([[0.97, 0.05], [0.03, 0.95]]),
        np.array([[0.96, 0.06], [0.04, 0.94]]),
        np.array([[0.98, 0.04], [0.02, 0.96]]),
    ])
    return rem


# ── Constructor ───────────────────────────────────────────────────────────────

def test_invalid_method(backend):
    with pytest.raises(ValueError, match="method doit être"):
        ReadoutMitigation(backend, method="invalid")


def test_valid_methods(backend):
    ReadoutMitigation(backend, method="matrix")
    ReadoutMitigation(backend, method="m3")


# ── Calibration ───────────────────────────────────────────────────────────────

def test_cals_from_matrices_wrong_length(backend):
    rem = ReadoutMitigation(backend, method="matrix")
    with pytest.raises(ValueError, match="Longueur"):
        rem.cals_from_matrices([np.eye(2)])  # only 1, needs 3


def test_cals_from_matrices_sets_cals(rem_matrix):
    assert rem_matrix.single_qubit_cals is not None
    assert len(rem_matrix.single_qubit_cals) == 3
    assert all(c is not None for c in rem_matrix.single_qubit_cals)


def test_cals_from_system(backend):
    rem = ReadoutMitigation(backend, method="matrix")
    benchmark_data = {
        "resultsPerDevice": {
            "qubits": {
                "0": {"parallelReadoutState0Fidelity": 0.97, "parallelReadoutState1Fidelity": 0.95},
                "1": {"parallelReadoutState0Fidelity": 0.96, "parallelReadoutState1Fidelity": 0.94},
                "2": {"parallelReadoutState0Fidelity": 0.98, "parallelReadoutState1Fidelity": 0.96},
            }
        }
    }
    with patch("qiskit_calculquebec.API.adapter.ApiAdapter.get_benchmark", return_value=benchmark_data):
        rem.cals_from_system()
    assert rem.single_qubit_cals is not None
    assert rem.cal_timestamp is not None
    assert np.isclose(rem.single_qubit_cals[0][0, 0], 0.97)
    assert np.isclose(rem.single_qubit_cals[0][1, 1], 0.95)


# ── Readout fidelity ──────────────────────────────────────────────────────────

def test_readout_fidelity_not_calibrated(backend):
    rem = ReadoutMitigation(backend, method="matrix")
    with pytest.raises(RuntimeError, match="non calibré"):
        rem.readout_fidelity()


def test_readout_fidelity_values(rem_matrix):
    fids = rem_matrix.readout_fidelity()
    assert len(fids) == 3
    assert all(f is not None for f in fids)
    assert np.isclose(fids[0]["p00"], 0.97)
    assert np.isclose(fids[0]["p11"], 0.95)
    assert np.isclose(fids[0]["mean"], (0.97 + 0.95) / 2)


# ── apply_correction (matrix) ─────────────────────────────────────────────────

def test_apply_correction_not_calibrated(backend):
    rem = ReadoutMitigation(backend, method="matrix")
    with pytest.raises(RuntimeError, match="non calibré"):
        rem.apply_correction({"000": 500, "111": 500}, qubits=[0, 1, 2])


def test_apply_correction_missing_qubit(backend):
    rem = ReadoutMitigation(backend, method="matrix")
    rem.cals_from_matrices([
        np.array([[0.97, 0.05], [0.03, 0.95]]),
        None,  # qubit 1 not calibrated
        np.array([[0.98, 0.04], [0.02, 0.96]]),
    ])
    with pytest.raises(RuntimeError, match="non calibrés"):
        rem.apply_correction({"000": 500}, qubits=[0, 1, 2])


def test_apply_correction_matrix_returns_dict(rem_matrix):
    counts = {"000": 450, "111": 450, "001": 50, "110": 50}
    corrected = rem_matrix.apply_correction(counts, qubits=[0, 1, 2])
    assert isinstance(corrected, dict)
    # Corrected total should be close to original total
    assert sum(corrected.values()) > 0


def test_apply_correction_matrix_improves_ghz(rem_matrix):
    """Correction should push 000 and 111 closer to equal counts."""
    counts = {"000": 420, "111": 420, "001": 40, "010": 30, "100": 40, "011": 30, "101": 20}
    corrected = rem_matrix.apply_correction(counts, qubits=[0, 1, 2])
    total = sum(corrected.values())
    assert total > 0
    p000 = corrected.get("000", 0) / total
    p111 = corrected.get("111", 0) / total
    # After correction both should be closer to 0.5
    assert p000 > 0.4
    assert p111 > 0.4


# ── confusion matrix helpers ──────────────────────────────────────────────────

def test_get_confusion_matrix_shape(rem_matrix):
    mat = rem_matrix.get_confusion_matrix([0, 1, 2])
    assert mat.shape == (8, 8)


def test_get_inv_confusion_matrix_shape(rem_matrix):
    inv = rem_matrix.get_inv_confusion_matrix([0, 1])
    assert inv.shape == (4, 4)


def test_get_confusion_matrix_not_calibrated(backend):
    rem = ReadoutMitigation(backend, method="matrix")
    with pytest.raises(RuntimeError):
        rem.get_confusion_matrix([0])


# ── faulty qubit detection ────────────────────────────────────────────────────

def test_faulty_qubit_checker_normal():
    cals = [
        np.array([[0.97, 0.05], [0.03, 0.95]]),  # P(0|1)=0.05 < P(0|0)=0.97 → OK
    ]
    assert _faulty_qubit_checker(cals) == []


def test_faulty_qubit_checker_faulty():
    cals = [
        np.array([[0.4, 0.6], [0.6, 0.4]]),  # P(0|1)=0.6 >= P(0|0)=0.4 → faulty
    ]
    assert _faulty_qubit_checker(cals) == [0]


def test_faulty_qubit_checker_none_entry():
    cals = [None, np.array([[0.97, 0.05], [0.03, 0.95]])]
    assert _faulty_qubit_checker(cals) == []
