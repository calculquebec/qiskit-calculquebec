from unittest.mock import MagicMock, patch
import pytest
from qiskit.circuit.library import *

from qiskit_calculquebec.API.client import CalculQuebecClient
from qiskit_calculquebec.backends.targets.monarq import MonarQ

client = CalculQuebecClient("host", "user", "token", project_id="test_project_id")


@pytest.fixture
def monarq_target():
    """Return a MonarQ instance with API calls mocked."""
    with patch(
        "qiskit_calculquebec.API.adapter.ApiAdapter.instance", autospec=True
    ) as mock_instance, patch(
        "qiskit_calculquebec.API.adapter.ApiAdapter.get_benchmark", autospec=True
    ) as mock_get_benchmark:

        # Make instance() return something truthy so Yukon goes into the API path
        mock_instance.return_value = MagicMock(name="ApiAdapterSingleton")

        mock_get_benchmark.return_value = {
            "resultsPerDevice": {
                "qubits": {
                    str(i): {
                        "t1": 10.0 + i,
                        "t2Echo": 20.0 + i,
                        # Optional fidelities; if omitted, Yukon uses defaults
                        "parallelSingleQubitGateFidelity": 0.999,
                        "parallelReadoutState1Fidelity": 0.98,
                    }
                    for i in range(24)
                },
                "couplers": {str(idx): {"czGateFidelity": 0.98} for idx in range(52)},
            }
        }

        yield MonarQ()


def test_qubit_count(monarq_target):
    assert monarq_target.num_qubits == 24


def test_single_qubit_gates(monarq_target):
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
            isinstance(instr, gate_cls) for instr, qubits in monarq_target.instructions
        )


def test_two_qubit_gates(monarq_target):
    assert any(
        isinstance(instr, CZGate) for instr, qubits in monarq_target.instructions
    )


def test_parameterized_gates_have_parameters(monarq_target):
    param_gates = [RZGate, PhaseGate]
    for gate_cls in param_gates:
        found = False
        for instr, qubits in monarq_target.instructions:
            if isinstance(instr, gate_cls):
                found = True
                assert len(instr.params) > 0
        assert found, f"{gate_cls.__name__} not found"


def test_coupling_map(monarq_target):
    expected_coupling_map = [
        (0, 4),
        (4, 0),
        (1, 4),
        (4, 1),
        (1, 5),
        (5, 1),
        (2, 5),
        (5, 2),
        (2, 6),
        (6, 2),
        (3, 6),
        (6, 3),
        (3, 7),
        (7, 3),
        (4, 8),
        (8, 4),
        (4, 9),
        (9, 4),
        (5, 9),
        (9, 5),
        (5, 10),
        (10, 5),
        (6, 10),
        (10, 6),
        (6, 11),
        (11, 6),
        (7, 11),
        (11, 7),
        (8, 12),
        (12, 8),
        (9, 12),
        (12, 9),
        (9, 13),
        (13, 9),
        (10, 13),
        (13, 10),
        (10, 14),
        (14, 10),
        (11, 14),
        (14, 11),
        (11, 15),
        (15, 11),
        (12, 16),
        (16, 12),
        (12, 17),
        (17, 12),
        (13, 17),
        (17, 13),
        (13, 18),
        (18, 13),
        (14, 18),
        (18, 14),
        (14, 19),
        (19, 14),
        (15, 19),
        (19, 15),
        (16, 20),
        (20, 16),
        (17, 20),
        (20, 17),
        (17, 21),
        (21, 17),
        (18, 21),
        (21, 18),
        (18, 22),
        (22, 18),
        (19, 22),
        (22, 19),
        (19, 23),
        (23, 19),
    ]
    assert monarq_target.coupling_map == expected_coupling_map
    assert len(monarq_target.coupling_map) == 70


def test_name(monarq_target):
    assert monarq_target.name == "MonarQ"


def test_qubit_properties(monarq_target):
    from qiskit.transpiler.target import QubitProperties

    qubit_props, gate_properties = monarq_target.__get_qubit_properties__()
    assert isinstance(qubit_props, list)
    assert isinstance(qubit_props[0], QubitProperties)
