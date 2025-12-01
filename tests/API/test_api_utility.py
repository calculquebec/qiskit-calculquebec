import pytest
from unittest.mock import patch
import numpy as np
from base64 import b64encode
from qiskit.circuit import Gate, QuantumCircuit
from qiskit_calculquebec.API.api_utility import (
    ApiUtility,
    keys,
)


def test_convert_instruction_supported_gate():
    # Create a dummy single-qubit gate
    class DummyGate(Gate):
        def __init__(self, name="x", num_qubits=1):
            super().__init__(name, num_qubits, [])

    gate = DummyGate("x")
    gate.qubits = [type("Qubit", (), {"_index": 0})()]

    result = ApiUtility.convert_instruction(gate)
    assert result[keys.TYPE] == "x"
    assert result[keys.QUBITS] == [0]


def test_convert_instruction_with_params():
    class DummyGate(Gate):
        def __init__(self, name="rz", num_qubits=1, param=np.pi):
            super().__init__(name, num_qubits, [param])
            self.params = [param]

    gate = DummyGate()
    gate.qubits = [type("Qubit", (), {"_index": 1})()]
    result = ApiUtility.convert_instruction(gate)
    assert result[keys.TYPE] == "rz"
    assert result[keys.QUBITS] == [1]
    assert abs(result[keys.PARAMETERS]["lambda"] - np.pi) < 1e-8


def test_convert_instruction_measure():
    class DummyGate(Gate):
        def __init__(self):
            super().__init__("measure", 1, [])

    gate = DummyGate()
    gate.qubits = [type("Qubit", (), {"_index": 0})()]
    gate.clbits = [type("Clbit", (), {"_index": 1})()]

    result = ApiUtility.convert_instruction(gate)
    assert result[keys.TYPE] == "readout"
    assert result[keys.QUBITS] == [0]
    assert result[keys.BITS] == [1]


def test_convert_instruction_unsupported():
    class DummyGate(Gate):
        def __init__(self):
            super().__init__("unsupported", 1, [])

    gate = DummyGate()
    gate.qubits = [type("Qubit", (), {"_index": 0})()]
    with pytest.raises(ValueError):
        ApiUtility.convert_instruction(gate)


def test_convert_circuit(monkeypatch):
    # Patch convert_instruction to track calls
    monkeypatch.setattr(ApiUtility, "convert_instruction", lambda x: {"dummy": x.name})

    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.cz(0, 1)

    result = ApiUtility.convert_circuit(circuit)
    assert result[keys.QUBIT_COUNT] == 2
    assert len(result[keys.OPERATIONS]) == 2
    assert all("dummy" in op for op in result[keys.OPERATIONS])


def test_basic_auth():
    token = ApiUtility.basic_auth("user", "password")
    assert token == "Basic " + b64encode(b"user:password").decode("ascii")


def test_headers(monkeypatch):
    monkeypatch.setattr(ApiUtility, "basic_auth", lambda u, p: "AUTH")
    result = ApiUtility.headers("user", "pass", "realm")
    assert result["Authorization"] == "AUTH"
    assert result["Content-Type"] == "application/json"
    assert result["X-Realm"] == "realm"


def test_job_body():
    circuit_dict = {"circuit": "data"}
    body = ApiUtility.job_body(circuit_dict, "name", "proj", "machine", 100)
    assert body[keys.CIRCUIT] == circuit_dict
    assert body[keys.NAME] == "name"
    assert body[keys.PROJECT_ID] == "proj"
    assert body[keys.MACHINE_NAME] == "machine"
    assert body[keys.SHOT_COUNT] == 100
