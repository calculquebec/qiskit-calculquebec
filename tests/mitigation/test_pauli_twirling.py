"""Tests for PauliTwirlingMitigation."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from qiskit import QuantumCircuit

from qiskit_calculquebec.mitigation.pauli_twirling import PauliTwirlingMitigation


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def backend():
    return MagicMock()


@pytest.fixture
def circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


@pytest.fixture
def mock_sampler_counts():
    counts = {"00": 900, "11": 100}
    result_mock = MagicMock()
    result_mock.join_data.return_value.get_counts.return_value = counts
    job_mock = MagicMock()
    job_mock.result.return_value = [result_mock]
    sampler_mock = MagicMock()
    sampler_mock.run.return_value = job_mock

    with patch("qiskit_calculquebec.mitigation.pauli_twirling.SamplerV2", return_value=sampler_mock), \
         patch("qiskit_calculquebec.mitigation.pauli_twirling.generate_preset_pass_manager") as pm_mock:
        pm_mock.return_value.run.side_effect = lambda c: c
        yield sampler_mock


# ── Constructor ───────────────────────────────────────────────────────────────

def test_defaults(backend):
    pt = PauliTwirlingMitigation(backend)
    assert pt.num_variants == 10
    assert pt.shots == 1024


def test_custom_params(backend):
    pt = PauliTwirlingMitigation(backend, num_variants=5, shots=512)
    assert pt.num_variants == 5
    assert pt.shots == 512


# ── Executor type dispatch ────────────────────────────────────────────────────

def test_base_executor_no_annotation(backend):
    """Base executor must have no return annotation (FloatLike for mitiq)."""
    import inspect
    pt = PauliTwirlingMitigation(backend)
    executor = pt._make_base_executor()
    ann = inspect.getfullargspec(executor).annotations
    assert ann.get("return") is None


def test_pt_executor_no_annotation(backend):
    """PT executor must have no return annotation."""
    import inspect
    pt = PauliTwirlingMitigation(backend)
    executor = pt._make_pt_executor()
    ann = inspect.getfullargspec(executor).annotations
    assert ann.get("return") is None


# ── run_unmitigated ───────────────────────────────────────────────────────────

def test_run_unmitigated_returns_float(backend, circuit, mock_sampler_counts):
    pt = PauliTwirlingMitigation(backend, shots=1000)
    result = pt.run_unmitigated(circuit)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


# ── run ───────────────────────────────────────────────────────────────────────

def test_run_averages_variants(backend, circuit):
    """run() should average over num_variants executions."""
    call_count = {"n": 0}

    def fake_base_executor(c):
        call_count["n"] += 1
        return 0.9

    with patch.object(PauliTwirlingMitigation, "_make_base_executor", return_value=fake_base_executor), \
         patch("mitiq.pt.generate_pauli_twirl_variants") as mock_variants:
        mock_variants.return_value = [circuit] * 3
        pt = PauliTwirlingMitigation(backend, num_variants=3)
        result = pt.run(circuit)

    assert call_count["n"] == 3
    assert isinstance(result, float)
    assert np.isclose(result, 0.9)


def test_run_returns_mean(backend, circuit):
    """run() should return the mean of variant results."""
    values = [0.8, 0.85, 0.9]
    idx = {"i": 0}

    def fake_base_executor(c):
        val = values[idx["i"]]
        idx["i"] += 1
        return val

    with patch.object(PauliTwirlingMitigation, "_make_base_executor", return_value=fake_base_executor), \
         patch("mitiq.pt.generate_pauli_twirl_variants") as mock_variants:
        mock_variants.return_value = [circuit] * 3
        pt = PauliTwirlingMitigation(backend, num_variants=3)
        result = pt.run(circuit)

    assert np.isclose(result, np.mean(values))


# ── run_with_zne ──────────────────────────────────────────────────────────────

def test_run_with_zne_uses_linear_factory_by_default(backend, circuit):
    from mitiq.zne.inference import LinearFactory
    captured = {}

    def fake_execute_with_zne(c, executor, **kwargs):
        captured["factory"] = kwargs.get("factory")
        return 0.9

    with patch("mitiq.zne.execute_with_zne", side_effect=fake_execute_with_zne), \
         patch("qiskit_calculquebec.mitigation.pauli_twirling.generate_preset_pass_manager"), \
         patch("qiskit_calculquebec.mitigation.pauli_twirling.SamplerV2"):
        pt = PauliTwirlingMitigation(backend)
        pt.run_with_zne(circuit)

    assert isinstance(captured["factory"], LinearFactory)


def test_run_with_zne_uses_custom_factory(backend, circuit):
    from mitiq.zne.inference import RichardsonFactory
    factory = RichardsonFactory([1.0, 2.0, 3.0])
    captured = {}

    def fake_execute_with_zne(c, executor, **kwargs):
        captured["factory"] = kwargs.get("factory")
        return 0.9

    with patch("mitiq.zne.execute_with_zne", side_effect=fake_execute_with_zne), \
         patch("qiskit_calculquebec.mitigation.pauli_twirling.generate_preset_pass_manager"), \
         patch("qiskit_calculquebec.mitigation.pauli_twirling.SamplerV2"):
        pt = PauliTwirlingMitigation(backend)
        pt.run_with_zne(circuit, factory=factory)

    assert captured["factory"] is factory


def test_run_with_zne_strips_measurements(backend, circuit):
    """Circuit passed to execute_with_zne must have no measurements."""
    captured = {}

    def fake_execute_with_zne(c, executor, **kwargs):
        captured["circuit"] = c
        return 0.9

    with patch("mitiq.zne.execute_with_zne", side_effect=fake_execute_with_zne), \
         patch("qiskit_calculquebec.mitigation.pauli_twirling.generate_preset_pass_manager"), \
         patch("qiskit_calculquebec.mitigation.pauli_twirling.SamplerV2"):
        pt = PauliTwirlingMitigation(backend)
        pt.run_with_zne(circuit)

    assert captured["circuit"].num_clbits == 0


def test_run_with_zne_returns_real_float(backend, circuit):
    with patch("mitiq.zne.execute_with_zne", return_value=complex(0.88, -1e-17)), \
         patch("qiskit_calculquebec.mitigation.pauli_twirling.generate_preset_pass_manager"), \
         patch("qiskit_calculquebec.mitigation.pauli_twirling.SamplerV2"):
        pt = PauliTwirlingMitigation(backend)
        result = pt.run_with_zne(circuit)

    assert isinstance(result, float)
    assert np.isclose(result, 0.88)


# ── run_variants ──────────────────────────────────────────────────────────────

def test_run_variants_length(backend, circuit, mock_sampler_counts):
    with patch("mitiq.pt.generate_pauli_twirl_variants") as mock_variants:
        mock_variants.return_value = [circuit] * 5
        pt = PauliTwirlingMitigation(backend, num_variants=5)
        results = pt.run_variants(circuit)

    assert len(results) == 5
    assert all(isinstance(v, float) for v in results)


# ── REM integration ───────────────────────────────────────────────────────────

def test_run_raises_rem_without_qubits(backend, circuit, mock_sampler_counts):
    rem = MagicMock()
    rem.method = "m3"
    pt = PauliTwirlingMitigation(backend)
    with patch("mitiq.pt.generate_pauli_twirl_variants") as mock_variants:
        mock_variants.return_value = [circuit]
        with pytest.raises(ValueError, match="qubits is required when rem is provided"):
            pt.run(circuit, rem=rem)


# ── Count key normalization ───────────────────────────────────────────────────

def test_count_key_normalization(backend):
    """Keys with spaces ('0 0') should be normalized to '00'."""
    counts = {"0 0": 900, "1 1": 100}
    result_mock = MagicMock()
    result_mock.join_data.return_value.get_counts.return_value = counts
    job_mock = MagicMock()
    job_mock.result.return_value = [result_mock]
    sampler_mock = MagicMock()
    sampler_mock.run.return_value = job_mock

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure_all()

    with patch("qiskit_calculquebec.mitigation.pauli_twirling.SamplerV2", return_value=sampler_mock), \
         patch("qiskit_calculquebec.mitigation.pauli_twirling.generate_preset_pass_manager") as pm_mock:
        pm_mock.return_value.run.side_effect = lambda c: c
        pt = PauliTwirlingMitigation(backend, shots=1000)
        executor = pt._make_base_executor()
        result = executor(qc)

    assert np.isclose(result, 900 / 1000)
