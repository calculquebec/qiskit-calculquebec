from typing import Dict
from unittest.mock import MagicMock, patch
import pytest
from qiskit.circuit.library import *

from qiskit_calculquebec.API.client import CalculQuebecClient
from qiskit_calculquebec.backends.targets.yukon import (
    Yukon,
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


@pytest.fixture
def yukon_target():
    return Yukon()


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


def test_qubit_properties(yukon_target, mock_api_adapter):
    from qiskit.transpiler.target import QubitProperties

    qubit_props = yukon_target.__get_qubit_properties__()
    assert isinstance(qubit_props, list)
    assert isinstance(qubit_props[0], QubitProperties)
