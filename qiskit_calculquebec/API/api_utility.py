"""
Contains API utility functions and constants
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Gate

# from qiskit.circuit.measurement import Measure
import numpy as np
from base64 import b64encode


class ApiUtility:
    @staticmethod
    def convert_instruction(instruction: Gate) -> dict:
        """Converts a Qiskit Gate to a dictionary that can be read by the Thunderhead API.

        Args:
            instruction (Gate): a Qiskit gate object.

        Returns:
            dict: a dictionary representation of the operation that can be read by the Thunderhead API.
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
            raise ValueError("This instruction is not supported: ", instruction.name)

        return operation

    @staticmethod
    def convert_circuit(circuit: QuantumCircuit) -> dict[str, any]:
        """Converts a QuantumCircuit to a dictionary that can be read by the Thunderhead API.

        Args:
            circuit (QuantumCircuit): A Qiskit quantum circuit (with information about the number of wires,
                                    operations, and measurements).

        Returns:
            dict[str, any]: A dictionary representation of the circuit that can be read by the API.
        """

        # Initialize the dictionary with fixed bit and qubit counts (adjustable as needed)
        # print(len(circuit))
        circuit_dict = {
            keys.TYPE: keys.CIRCUIT,
            keys.BIT_COUNT: 24,  # Adjust as needed for dynamic sizing
            keys.OPERATIONS: [
                ApiUtility.convert_instruction(operation) for operation in circuit.data
            ],
            keys.QUBIT_COUNT: len(circuit.qubits),  # Number of qubits in the circuit
        }

        return circuit_dict

    @staticmethod
    def basic_auth(username: str, password: str) -> str:
        """create a basic authentication token from a Thunderhead username and access token

        Args:
            username (str): your Thunderhead username
            password (str): your Thunderhead access token

        Returns:
            str: the basic authentification string that will authenticate you with the API
        """
        token = b64encode(f"{username}:{password}".encode("ascii")).decode("ascii")
        return f"Basic {token}"

    @staticmethod
    def headers(username: str, password: str, realm: str) -> dict[str, str]:
        """the Thunderhead API headers

        Args:
            username (str): your Thunderhead username
            password (str): your Thunderhead access token
            realm (str): your organization identifier with Thunderhead

        Returns:
            dict[str, any]: a dictionary representing the request headers
        """
        return {
            "Authorization": ApiUtility.basic_auth(username, password),
            "Content-Type": "application/json",
            "X-Realm": realm,
        }

    @staticmethod
    def job_body(
        circuit: dict[str, any],
        circuit_name: str,
        project_id: str,
        machine_name: str,
        shots,
    ) -> dict[str, any]:
        """the body for the job creation request

        Args:
            circuit (tape.QuantumScript): the script you want to convert
            name (str): the name of your job
            project_id (str): the id for the project for which this job will be run
            machine_name (str): the name of the machine on which this job will be run
            shots (int, optional): the number of shots (-1 will use the circuit's shot number)

        Returns:
            dict[str, any]: the body for the job creation request
        """
        body = {
            keys.NAME: circuit_name,
            keys.PROJECT_ID: project_id,
            keys.MACHINE_NAME: machine_name,
            keys.SHOT_COUNT: shots,
            keys.CIRCUIT: circuit,
        }
        return body


class routes:
    JOBS = "/jobs"
    PROJECTS = "/projects"
    MACHINES = "/machines"
    BENCHMARKING = "/benchmarking"

class queries:
    MACHINE_NAME = "?machineName"
    NAME = "?name"

class keys:
    BIT_COUNT = "bitCount"
    QUBIT_COUNT = "qubitCount"
    OPERATIONS = "operations"
    CIRCUIT = "circuit"
    NAME = "name"
    MACHINE_NAME = "machineName"
    PROJECT_ID = "projectID"
    SHOT_COUNT = "shotCount"
    TYPE = "type"
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


instructions: dict[str, str] = {
    "i": "i",
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

instructions_with_params: dict[str, str] = {"rz": "rz", "p": "p"}
