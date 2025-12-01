import pytest
from unittest.mock import patch, MagicMock
from qiskit_calculquebec.API.client import CalculQuebecClient
from qiskit_calculquebec.backends.monarq_backend import MonarQBackend
from qiskit_calculquebec.backends.utils.job import MultiMonarQJob


client = CalculQuebecClient("host", "user", "token", project_id="test_project_id")


@pytest.fixture
def mock_api_adapter():
    with patch("qiskit_calculquebec.API.adapter.ApiAdapter.instance") as mock_instance:
        mock_instance.return_value = MagicMock(client=client)

        # Mock get_benchmark to return the expected dict
        benchmark_data = {
            "resultsPerDevice": {
                "qubits": {
                    str(i): {"t1": 10.0 + i, "t2Echo": 20.0 + i} for i in range(6)
                }
            }
        }
        with patch(
            "qiskit_calculquebec.API.adapter.ApiAdapter.get_benchmark",
            return_value=benchmark_data,
        ):
            with patch(
                "qiskit_calculquebec.API.adapter.ApiAdapter.get_machine_by_name"
            ) as mock_machine:
                with patch(
                    "qiskit_calculquebec.API.adapter.ApiAdapter.post_job"
                ) as mock_post_job:
                    yield mock_instance, MagicMock(), mock_machine, mock_post_job


def test_constructor(mock_api_adapter):
    mock_instance, mock_bench, mock_machine, mock_post_job = mock_api_adapter
    # no client given, should raise ValueError
    with pytest.raises(ValueError):
        dev = MonarQBackend()

    mock_instance.assert_not_called()

    # client given, no config given, should set default config
    dev = MonarQBackend(client=client)
    mock_instance.assert_called_once()
    assert dev.name.lower() in ["yukon", "monarq"]


def test_validate_circuit(mock_api_adapter):
    mock_instance, mock_bench, mock_machine, mock_post_job = mock_api_adapter
    dev = MonarQBackend(client=client)
    mock_instance.assert_called_once()

    # valid circuit
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    dev._validate_circuit([qc])  # should not raise

    # invalid circuit (no measurement)
    qc2 = QuantumCircuit(2)
    qc2.h(0)
    qc2.cx(0, 1)

    with pytest.raises(ValueError):
        dev._validate_circuit([qc2])

    # invalid circuit (gate after measurement)
    qc2 = QuantumCircuit(2)
    qc2.h(0)
    qc2.cx(0, 1)
    qc2.measure_all()
    qc2.x(0)

    with pytest.raises(ValueError):
        dev._validate_circuit([qc2])


def test_default_options(mock_api_adapter):
    mock_instance, mock_bench, mock_machine, mock_post_job = mock_api_adapter
    dev = MonarQBackend(client=client)
    mock_instance.assert_called_once()

    options = dev._default_options()
    assert options.shots == 1024


def test_run_sets_shots_limit(mock_api_adapter):

    dev = MonarQBackend(client=client)

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.x(0)
    qc.cz(0, 1)
    qc.measure_all()

    # Run with shots greater than 1024
    with pytest.warns(Warning):
        job = dev.run([qc], shots=2000)
        assert job.shots == 1024

    # Run with shots less than or equal to 1024
    job = dev.run([qc], shots=500)
    assert job.shots == 500


def test_run_multiple_circuits(mock_api_adapter):

    dev = MonarQBackend(client=client)

    from qiskit import QuantumCircuit

    qc1 = QuantumCircuit(2)
    qc1.x(0)
    qc1.cz(0, 1)
    qc1.measure_all()

    qc2 = QuantumCircuit(3)
    qc2.x(0)
    qc2.cz(0, 1)
    qc2.cz(1, 2)
    qc2.measure_all()

    job = dev.run([qc1, qc2], shots=800)
    assert job.shots == 800
    assert len(job.circuits) == 2
    assert isinstance(job, MultiMonarQJob)
