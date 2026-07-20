"""Tests for ZNEMitigation."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call
from qiskit import QuantumCircuit

from qiskit_calculquebec.mitigation.zne import ZNEMitigation

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def backend():
    mock = MagicMock()
    mock.target.num_qubits = 3
    return mock


@pytest.fixture
def ghz():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    return qc


@pytest.fixture
def mock_sampler_counts():
    """Patch SamplerV2 to return a fixed counts dict."""
    counts = {"000": 480, "111": 490, "001": 10, "110": 20}

    result_mock = MagicMock()
    result_mock.join_data.return_value.get_counts.return_value = counts

    job_mock = MagicMock()
    job_mock.result.return_value = [result_mock]

    sampler_mock = MagicMock()
    sampler_mock.run.return_value = job_mock

    with patch(
        "qiskit_calculquebec.mitigation.zne.SamplerV2", return_value=sampler_mock
    ), patch(
        "qiskit_calculquebec.mitigation.zne.generate_preset_pass_manager"
    ) as pm_mock:
        pm_mock.return_value.run.side_effect = lambda c: c
        yield sampler_mock


# ── Constructor defaults ──────────────────────────────────────────────────────


def test_default_scale_factors(backend):
    zne = ZNEMitigation(backend)
    assert zne.scale_factors == [1.0, 1.5, 2.0, 2.5, 3.0]


def test_custom_scale_factors(backend):
    zne = ZNEMitigation(backend, scale_factors=[1.0, 2.0, 3.0])
    assert zne.scale_factors == [1.0, 2.0, 3.0]


def test_default_shots(backend):
    zne = ZNEMitigation(backend)
    assert zne.shots == 1024


# ── Executor type dispatch ────────────────────────────────────────────────────


def test_executor_float_mode_no_annotation(backend):
    """Float executor must have no return annotation so mitiq treats it as FloatLike."""
    import inspect

    zne = ZNEMitigation(backend)
    executor = zne._make_executor()
    ann = inspect.getfullargspec(executor).annotations
    assert ann.get("return") is None


def test_executor_measurement_result_mode_annotation(backend):
    """MeasurementResult executor must have correct annotation for mitiq dispatch."""
    import inspect
    from mitiq import MeasurementResult
    from mitiq import Observable, PauliString

    obs = Observable(PauliString("ZZ", support=[0, 1]))
    zne = ZNEMitigation(backend)
    executor = zne._make_executor(observable=obs)
    ann = inspect.getfullargspec(executor).annotations
    assert ann.get("return") is MeasurementResult


# ── run_unmitigated ───────────────────────────────────────────────────────────


def test_run_unmitigated_returns_float(backend, ghz, mock_sampler_counts):
    zne = ZNEMitigation(backend, shots=1000)
    result = zne.run_unmitigated(ghz)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_run_unmitigated_p000(backend, ghz, mock_sampler_counts):
    """Should return P(|000⟩) = 480/1024."""
    zne = ZNEMitigation(backend, shots=1024)
    result = zne.run_unmitigated(ghz)
    assert np.isclose(result, 480 / 1024)


# ── run ───────────────────────────────────────────────────────────────────────


def test_run_uses_linear_factory_by_default(backend, ghz):
    """Default factory should be LinearFactory, not Richardson."""
    from mitiq.zne.inference import LinearFactory

    zne = ZNEMitigation(backend, scale_factors=[1.0, 2.0, 3.0])
    captured = {}

    def fake_execute_with_zne(circuit, executor, **kwargs):
        captured["factory"] = kwargs.get("factory")
        return 0.5

    with patch("mitiq.zne.execute_with_zne", side_effect=fake_execute_with_zne), patch(
        "qiskit_calculquebec.mitigation.zne.generate_preset_pass_manager"
    ), patch("qiskit_calculquebec.mitigation.zne.SamplerV2"):
        zne.run(ghz)

    assert isinstance(captured["factory"], LinearFactory)


def test_run_uses_custom_factory(backend, ghz):
    from mitiq.zne.inference import RichardsonFactory

    factory = RichardsonFactory([1.0, 2.0, 3.0])
    zne = ZNEMitigation(backend, factory=factory)
    captured = {}

    def fake_execute_with_zne(circuit, executor, **kwargs):
        captured["factory"] = kwargs.get("factory")
        return 0.5

    with patch("mitiq.zne.execute_with_zne", side_effect=fake_execute_with_zne), patch(
        "qiskit_calculquebec.mitigation.zne.generate_preset_pass_manager"
    ), patch("qiskit_calculquebec.mitigation.zne.SamplerV2"):
        zne.run(ghz)

    assert captured["factory"] is factory


def test_run_strips_measurements(backend, ghz):
    """Circuit passed to execute_with_zne must have no measurements."""
    captured = {}

    def fake_execute_with_zne(circuit, executor, **kwargs):
        captured["circuit"] = circuit
        return 0.5

    with patch("mitiq.zne.execute_with_zne", side_effect=fake_execute_with_zne), patch(
        "qiskit_calculquebec.mitigation.zne.generate_preset_pass_manager"
    ), patch("qiskit_calculquebec.mitigation.zne.SamplerV2"):
        zne = ZNEMitigation(backend)
        zne.run(ghz)

    assert captured["circuit"].num_clbits == 0


def test_run_returns_real_float(backend, ghz):
    with patch("mitiq.zne.execute_with_zne", return_value=complex(0.85, -1e-17)), patch(
        "qiskit_calculquebec.mitigation.zne.generate_preset_pass_manager"
    ), patch("qiskit_calculquebec.mitigation.zne.SamplerV2"):
        zne = ZNEMitigation(backend)
        result = zne.run(ghz)

    assert isinstance(result, float)
    assert np.isclose(result, 0.85)


# ── REM integration ───────────────────────────────────────────────────────────


def test_run_unmitigated_raises_if_rem_without_qubits(
    backend, ghz, mock_sampler_counts
):
    rem = MagicMock()
    rem.method = "m3"
    zne = ZNEMitigation(backend)
    with pytest.raises(ValueError, match="qubits is required when rem is provided"):
        zne.run_unmitigated(ghz, rem=rem)


def test_run_unmitigated_applies_rem_matrix(backend, ghz):
    from qiskit_calculquebec.mitigation.readout import ReadoutMitigation
    import numpy as np

    rem = MagicMock(spec=ReadoutMitigation)
    rem.method = "matrix"
    rem.apply_correction.return_value = {"000": 490, "111": 510}

    counts = {"000": 480, "111": 490, "001": 10, "110": 20}
    result_mock = MagicMock()
    result_mock.join_data.return_value.get_counts.return_value = counts
    job_mock = MagicMock()
    job_mock.result.return_value = [result_mock]
    sampler_mock = MagicMock()
    sampler_mock.run.return_value = job_mock

    with patch(
        "qiskit_calculquebec.mitigation.zne.SamplerV2", return_value=sampler_mock
    ), patch(
        "qiskit_calculquebec.mitigation.zne.generate_preset_pass_manager"
    ) as pm_mock:
        pm_mock.return_value.run.side_effect = lambda c: c
        zne = ZNEMitigation(backend, shots=1024)
        result = zne.run_unmitigated(ghz, rem=rem, qubits=[0, 1, 2])

    rem.apply_correction.assert_called_once()
    assert isinstance(result, float)
