"""
Utility functions and constants for the Thunderhead REST API.

Provides circuit/instruction conversion, request header construction,
and string constants used across the API layer.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
import numpy as np
from base64 import b64encode


class ApiUtility:
    """Static helper methods for building Thunderhead API payloads."""

    @staticmethod
    def convert_instruction(instruction: Gate) -> dict:
        """Convert a Qiskit gate to a Thunderhead operation dictionary.

        Args:
            instruction (Gate): A Qiskit gate instance with ``qubits``,
                ``clbits``, and ``params`` attributes populated (as found in
                ``QuantumCircuit.data``).

        Returns:
            dict: Thunderhead operation dict with ``type``, ``qubits``, and
                optionally ``bits`` or ``parameters`` fields.

        Raises:
            ValueError: If the instruction name is not in the supported gate set.
        """
        if instruction.name in instructions:
            if len(instruction.qubits) == 1:
                operation = {
                    keys.QUBITS: [instruction.qubits[0]._index],
                    keys.TYPE: instructions[instruction.name],
                }
            else:
                operation = {
                    keys.TYPE: instructions[instruction.name],
                    keys.QUBITS: [
                        instruction.qubits[0]._index,
                        instruction.qubits[1]._index,
                    ],
                }

        elif instruction.name in instructions_with_params:
            value = (
                instruction.params[0]
                if isinstance(instruction.params[0], (int, float, np.ndarray))
                else instruction.params[0]
            )
            operation = {
                keys.TYPE: instructions_with_params[instruction.name],
                keys.QUBITS: [instruction.qubits[0]._index],
                keys.PARAMETERS: {"lambda": value},
            }

        elif instruction.name == "measure":
            operation = {
                keys.TYPE: "readout",
                keys.QUBITS: [instruction.qubits[0]._index],
                keys.BITS: [instruction.clbits[0]._index],
            }

        else:
            raise ValueError(f"Unsupported instruction: {instruction.name!r}")

        return operation

    @staticmethod
    def convert_circuit(circuit: QuantumCircuit) -> dict:
        """Convert a ``QuantumCircuit`` to the Thunderhead circuit dictionary format.

        Args:
            circuit (QuantumCircuit): Circuit to convert. All instructions must
                be in the supported gate set.

        Returns:
            dict: Thunderhead circuit dict with ``type``, ``bitCount``,
                ``qubitCount``, and ``operations`` fields.
        """
        return {
            keys.TYPE: keys.CIRCUIT,
            keys.BIT_COUNT: 24,
            keys.OPERATIONS: [
                ApiUtility.convert_instruction(op) for op in circuit.data
            ],
            keys.QUBIT_COUNT: len(circuit.qubits),
        }

    @staticmethod
    def basic_auth(username: str, password: str) -> str:
        """Build a Basic Authentication header value.

        Args:
            username (str): Thunderhead username.
            password (str): Thunderhead access token.

        Returns:
            str: ``"Basic <base64-encoded credentials>"`` header value.
        """
        token = b64encode(f"{username}:{password}".encode("ascii")).decode("ascii")
        return f"Basic {token}"

    @staticmethod
    def headers(username: str, password: str, realm: str) -> dict[str, str]:
        """Build the HTTP request headers required by the Thunderhead API.

        Args:
            username (str): Thunderhead username.
            password (str): Thunderhead access token.
            realm (str): Organizational realm identifier.

        Returns:
            dict[str, str]: Headers dict containing ``Authorization``,
                ``Content-Type``, and ``X-Realm``.
        """
        return {
            "Authorization": ApiUtility.basic_auth(username, password),
            "Content-Type": "application/json",
            "X-Realm": realm,
        }

    @staticmethod
    def job_body(
        circuit: dict,
        circuit_name: str,
        project_id: str,
        machine_name: str,
        shots: int,
    ) -> dict:
        """Build the request body for the ``POST /jobs`` endpoint.

        Args:
            circuit (dict): Circuit in Thunderhead dictionary format.
            circuit_name (str): Human-readable label for the circuit.
            project_id (str): ID of the project under which the job will be billed.
            machine_name (str): Target machine name (e.g. ``"yukon"``).
            shots (int): Number of shots to execute.

        Returns:
            dict: JSON-serializable body for the job creation request.
        """
        return {
            keys.NAME: circuit_name,
            keys.PROJECT_ID: project_id,
            keys.MACHINE_NAME: machine_name,
            keys.SHOT_COUNT: shots,
            keys.CIRCUIT: circuit,
        }


class routes:
    """URL path segments for Thunderhead API endpoints."""

    JOBS = "/jobs"
    PROJECTS = "/projects"
    MACHINES = "/machines"
    BENCHMARKING = "/benchmarking"


class queries:
    """Query parameter prefixes used in Thunderhead API requests."""

    MACHINE_NAME = "?name"
    NAME = "?name"


class keys:
    """JSON field name constants used in Thunderhead request/response payloads."""

    NAME = "name"
    STATUS = "status"
    ONLINE = "online"
    COUPLER_TO_QUBIT_MAP = "couplerToQubitMap"
    BIT_COUNT = "bitCount"
    TYPE = "type"
    QUBIT_COUNT = "qubitCount"
    OPERATIONS = "operations"
    CIRCUIT = "circuit"
    MACHINE_NAME = "name"
    PROJECT_ID = "projectID"
    SHOT_COUNT = "shotCount"
    BITS = "bits"
    QUBITS = "qubits"
    PARAMETERS = "parameters"
    COUPLERS = "couplers"
    SINGLE_QUBIT_GATE_FIDELITY = "singleQubitGateFidelity"
    READOUT_STATE_0_FIDELITY = "readoutState0Fidelity"
    READOUT_STATE_1_FIDELITY = "readoutState1Fidelity"
    T1 = "t1"
    T2_RAMSEY = "t2Ramsey"
    CZ_GATE_FIDELITY = "czGateFidelity"
    RESULTS_PER_DEVICE = "resultsPerDevice"
    ITEMS = "items"
    ID = "id"


# Gate names that map directly to Thunderhead instruction types (no parameters)
instructions: dict[str, str] = {
    "i": "i",
    "id": "i",
    "x": "x",
    "y": "y",
    "z": "z",
    "t": "t",
    "tdg": "t_dag",
    "sx": "x_90",
    "sxdg": "x_minus_90",
    "ry90": "y_90",
    "rym90": "y_minus_90",
    "cz": "cz",
}

# Gate names that require a rotation angle parameter
instructions_with_params: dict[str, str] = {"rz": "rz", "p": "p"}
