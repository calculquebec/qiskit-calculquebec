from typing import Dict
from unittest.mock import MagicMock, patch
import pytest
from qiskit.circuit.library import *

from qiskit_calculquebec.API.client import CalculQuebecClient
from qiskit_calculquebec.backends.targets.monarq import (
    MonarQ,
)


client = CalculQuebecClient("host", "user", "token", project_id="test_project_id")


@pytest.fixture
def mock_api_adapter():
    with patch("qiskit_calculquebec.API.adapter.ApiAdapter.instance") as mock_instance:
        mock_instance.return_value = MagicMock(client=client)

        # Mock get_benchmark to return the expected dict
        benchmark_data = {
            "resultsPerDevice": {
                "qubits": {
                    str(i): {"t1": 10.0 + i, "t2Echo": 20.0 + i} for i in range(24)
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


@pytest.fixture
def monarq_target():
    return MonarQ()


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
        (1, 5),
        (2, 6),
        (3, 7),
        (4, 0),
        (4, 1),
        (4, 8),
        (5, 1),
        (5, 2),
        (5, 9),
        (5, 10),
        (6, 2),
        (6, 3),
        (6, 11),
        (7, 3),
        (7, 13),
        (8, 4),
        (8, 12),
        (9, 4),
        (9, 5),
        (9, 13),
        (10, 5),
        (10, 10),
        (10, 14),
        (11, 6),
        (11, 12),
        (11, 14),
        (12, 8),
        (12, 11),
        (12, 17),
        (13, 7),
        (13, 9),
        (13, 18),
        (14, 10),
        (14, 11),
        (14, 19),
        (15, 11),
        (15, 19),
        (16, 12),
        (16, 20),
        (17, 12),
        (17, 16),
        (17, 21),
        (18, 13),
        (18, 22),
        (19, 14),
        (19, 15),
        (19, 23),
        (20, 16),
        (21, 17),
        (22, 18),
        (23, 19),
    ]
    assert monarq_target.coupling_map == expected_coupling_map


def test_name(monarq_target):
    assert monarq_target.name == "MonarQ"


def test_qubit_properties(monarq_target, mock_api_adapter):
    from qiskit.transpiler.target import QubitProperties

    qubit_props = monarq_target.__get_qubit_properties__()
    assert isinstance(qubit_props, list)
    assert isinstance(qubit_props[0], QubitProperties)
