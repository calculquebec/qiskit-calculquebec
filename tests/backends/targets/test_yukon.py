from unittest.mock import MagicMock, patch
import pytest
from qiskit.circuit.library import *

from qiskit_calculquebec.API.client import CalculQuebecClient
from qiskit_calculquebec.backends.targets.yukon import (
    Yukon,
)

client = CalculQuebecClient("host", "user", "token", project_id="test_project_id")


@pytest.fixture
def yukon_target():
    """Return a Yukon instance with API calls mocked."""
    with patch(
        "qiskit_calculquebec.API.adapter.ApiAdapter.instance", autospec=True
    ) as mock_instance, patch(
        "qiskit_calculquebec.API.adapter.ApiAdapter.get_benchmark", autospec=True
    ) as mock_get_benchmark:

        # Make instance() return something truthy so Yukon goes into the API path
        mock_instance.return_value = MagicMock(name="ApiAdapterSingleton")

        # Build exactly what Yukon.__get_qubit_properties__ expects
        mock_get_benchmark.return_value = {
            "resultsPerDevice": {
                "qubits": {
                    str(i): {
                        "t1": 10.0 + i,
                        "t2Echo": 20.0 + i,
                        "parallelSingleQubitGateFidelity": 0.999,
                        "parallelReadoutState1Fidelity": 0.98,
                    }
                    for i in range(6)
                },
                "couplers": {
                    # Yukon indexes couplers by str(idx) where idx enumerates coupling_map
                    str(idx): {"czGateFidelity": 0.98}
                    for idx in range(12)
                },
            }
        }

        yield Yukon()


def test_qubit_count(yukon_target):
    assert yukon_target.num_qubits == 6


def test_single_qubit_gates(yukon_target):
    gates_to_check = [
        IGate,
        XGate,
        YGate,
        ZGate,
        TGate,
        TdgGate,
        RZGate,
        PhaseGate,
        SXGate,
        SXdgGate,
        Measure,
    ]
    for gate_cls in gates_to_check:
        assert any(
            isinstance(instr, gate_cls) for instr, qubits in yukon_target.instructions
        )


def test_two_qubit_gates(yukon_target):
    assert any(isinstance(instr, CZGate) for instr, qubits in yukon_target.instructions)


def test_parameterized_gates_have_parameters(yukon_target):
    param_gates = [RZGate, PhaseGate]
    for gate_cls in param_gates:
        found = False
        for instr, qubits in yukon_target.instructions:
            if isinstance(instr, gate_cls):
                found = True
                assert len(instr.params) > 0
        assert found, f"{gate_cls.__name__} not found"


def test_coupling_map(yukon_target):
    expected_coupling_map = [
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
        (3, 4),
        (4, 3),
        (4, 5),
        (5, 4),
    ]
    assert yukon_target.coupling_map == expected_coupling_map


def test_name(yukon_target):
    assert yukon_target.name == "Yukon"


def test_qubit_properties(yukon_target):
    from qiskit.transpiler.target import QubitProperties

    qubit_props, gate_properties = yukon_target.__get_qubit_properties__()
    assert isinstance(qubit_props, list)
    assert isinstance(qubit_props[0], QubitProperties)
