"""Tests for DDDMitigation."""

import pytest
from unittest.mock import MagicMock, patch
from qiskit import QuantumCircuit

from qiskit_calculquebec.mitigation.ddd import DDDMitigation

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def backend():
    return MagicMock()


@pytest.fixture
def idle_circuit():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    for _ in range(5):
        qc.id(0)
        qc.id(1)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])
    return qc


@pytest.fixture
def mock_sampler_counts():
    counts = {"00": 900, "01": 50, "10": 30, "11": 20}
    result_mock = MagicMock()
    result_mock.join_data.return_value.get_counts.return_value = counts
    job_mock = MagicMock()
    job_mock.result.return_value = [result_mock]
    sampler_mock = MagicMock()
    sampler_mock.run.return_value = job_mock

    with patch(
        "qiskit_calculquebec.mitigation.ddd.SamplerV2", return_value=sampler_mock
    ), patch(
        "qiskit_calculquebec.mitigation.ddd.generate_preset_pass_manager"
    ) as pm_mock:
        pm_mock.return_value.run.side_effect = lambda c: c
        yield sampler_mock


# ── Constructor ───────────────────────────────────────────────────────────────


def test_invalid_rule(backend):
    with pytest.raises(ValueError, match="rule must be one of"):
        DDDMitigation(backend, rule="invalid")


def test_valid_rules(backend):
    for rule in ["xx", "yy", "xyxy"]:
        ddd = DDDMitigation(backend, rule=rule)
        assert ddd.rule == rule


def test_default_rule(backend):
    ddd = DDDMitigation(backend)
    assert ddd.rule == "xyxy"


def test_default_num_trials(backend):
    ddd = DDDMitigation(backend)
    assert ddd.num_trials == 3


# ── Executor type dispatch ────────────────────────────────────────────────────


def test_executor_float_no_annotation(backend):
    """Float executor must have no return annotation."""
    import inspect

    ddd = DDDMitigation(backend)
    executor = ddd._make_executor()
    ann = inspect.getfullargspec(executor).annotations
    assert ann.get("return") is None


def test_executor_measurement_result_annotation(backend):
    """MeasurementResult executor must be correctly annotated."""
    import inspect
    from mitiq import MeasurementResult, Observable, PauliString

    obs = Observable(PauliString("ZZ", support=[0, 1]))
    ddd = DDDMitigation(backend)
    executor = ddd._make_executor(observable=obs)
    ann = inspect.getfullargspec(executor).annotations
    assert ann.get("return") is MeasurementResult


# ── run_unmitigated ───────────────────────────────────────────────────────────


def test_run_unmitigated_returns_float(backend, idle_circuit, mock_sampler_counts):
    ddd = DDDMitigation(backend)
    result = ddd.run_unmitigated(idle_circuit)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_run_unmitigated_p00(backend, idle_circuit, mock_sampler_counts):
    """Should return P(|00⟩) = 900/1024."""
    ddd = DDDMitigation(backend, shots=1024)
    result = ddd.run_unmitigated(idle_circuit)
    assert abs(result - 900 / 1024) < 1e-9


# ── run ───────────────────────────────────────────────────────────────────────


def test_run_calls_execute_with_ddd(backend, idle_circuit):
    captured = {}

    def fake_execute_with_ddd(circuit, executor, rule, num_trials, **kwargs):
        captured["rule_name"] = (
            rule.__name__ if hasattr(rule, "__name__") else str(rule)
        )
        captured["num_trials"] = num_trials
        return 0.85

    with patch("mitiq.ddd.execute_with_ddd", side_effect=fake_execute_with_ddd), patch(
        "qiskit_calculquebec.mitigation.ddd.generate_preset_pass_manager"
    ), patch("qiskit_calculquebec.mitigation.ddd.SamplerV2"):
        ddd = DDDMitigation(backend, rule="xyxy", num_trials=5)
        result = ddd.run(idle_circuit)

    assert captured["num_trials"] == 5
    assert isinstance(result, float)


def test_run_strips_measurements_with_observable(backend, idle_circuit):
    from mitiq import Observable, PauliString

    obs = Observable(PauliString("ZZ", support=[0, 1]))
    captured = {}

    def fake_execute_with_ddd(circuit, executor, rule, num_trials, **kwargs):
        captured["circuit"] = circuit
        return 0.85

    with patch("mitiq.ddd.execute_with_ddd", side_effect=fake_execute_with_ddd), patch(
        "qiskit_calculquebec.mitigation.ddd.generate_preset_pass_manager"
    ), patch("qiskit_calculquebec.mitigation.ddd.SamplerV2"):
        ddd = DDDMitigation(backend)
        ddd.run(idle_circuit, observable=obs)

    assert captured["circuit"].num_clbits == 0


def test_run_returns_real_float(backend, idle_circuit):
    with patch("mitiq.ddd.execute_with_ddd", return_value=complex(0.75, -1e-18)), patch(
        "qiskit_calculquebec.mitigation.ddd.generate_preset_pass_manager"
    ), patch("qiskit_calculquebec.mitigation.ddd.SamplerV2"):
        ddd = DDDMitigation(backend)
        result = ddd.run(idle_circuit)

    assert isinstance(result, float)
    assert abs(result - 0.75) < 1e-10


# ── REM integration ───────────────────────────────────────────────────────────


def test_run_unmitigated_raises_rem_without_qubits(
    backend, idle_circuit, mock_sampler_counts
):
    rem = MagicMock()
    rem.method = "m3"
    ddd = DDDMitigation(backend)
    with pytest.raises(ValueError, match="qubits is required when rem is provided"):
        ddd.run_unmitigated(idle_circuit, rem=rem)


# ── Count key normalization ───────────────────────────────────────────────────


def test_count_key_normalization(backend):
    """Multi-register counts with spaces ('0 0') should be normalized."""
    counts = {"0 0": 900, "0 1": 50, "1 0": 30, "1 1": 20}
    result_mock = MagicMock()
    result_mock.join_data.return_value.get_counts.return_value = counts
    job_mock = MagicMock()
    job_mock.result.return_value = [result_mock]
    sampler_mock = MagicMock()
    sampler_mock.run.return_value = job_mock

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure_all()

    with patch(
        "qiskit_calculquebec.mitigation.ddd.SamplerV2", return_value=sampler_mock
    ), patch(
        "qiskit_calculquebec.mitigation.ddd.generate_preset_pass_manager"
    ) as pm_mock:
        pm_mock.return_value.run.side_effect = lambda c: c
        ddd = DDDMitigation(backend, shots=1000)
        executor = ddd._make_executor()
        result = executor(qc)

    assert isinstance(result, float)
    assert abs(result - 900 / 1000) < 1e-9
