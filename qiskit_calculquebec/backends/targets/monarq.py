from qiskit.transpiler.target import Target
from qiskit.circuit.library import (
    IGate,
    XGate,
    YGate,
    ZGate,
    TGate,
    TdgGate,
    PhaseGate,
    CZGate,
    RZGate,
    SXGate,
    SXdgGate,
    Measure,
)
from qiskit.circuit import Parameter
from qiskit.transpiler.target import QubitProperties

# Custom single-qubit rotations (not in standard Qiskit)
from qiskit_calculquebec.API.adapter import ApiAdapter
from qiskit_calculquebec.custom_gates.ry_90_gate import RY90Gate
from qiskit_calculquebec.custom_gates.ry_m90_gate import RYm90Gate


class MonarQ(Target):
    """
    Custom Qiskit Target for the Yukon 6-qubit device.

    Defines:
    - Qubit connectivity (coupling map)
    - Supported single- and two-qubit gates
    - Measurement operations
    """

    def __init__(self):
        super().__init__()
        qubit_properties = self.__get_qubit_properties__()
        self.qubit_properties = qubit_properties
        self.name = "MonarQ"

        # Define the bidirectional connectivity of the 6 qubits
        # Each tuple represents a directed edge (control, target)
        self.coupling_map = [
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

        # Define qubits (0 to 23)
        qubits = range(24)

        # Parameter for parameterized gates (RZ and Phase)
        phi = Parameter("φ")

        # --- Single-qubit gates ---
        single_qubit_gates = [
            IGate(),
            XGate(),
            YGate(),
            ZGate(),
            TGate(),
            TdgGate(),
            RZGate(phi),
            PhaseGate(phi),
            SXGate(),
            SXdgGate(),
            Measure(),
            RY90Gate(),
            RYm90Gate(),  # Custom gates
        ]

        # Add each single-qubit gate to all qubits
        for gate in single_qubit_gates:
            # Map gate to all qubits (key: tuple of qubit index)
            gate_props = {(q,): None for q in qubits}
            self.add_instruction(gate, gate_props)

        # --- Two-qubit gates ---
        # Only CZ is supported, defined for all edges in the coupling map
        cz_props = {edge: None for edge in self.coupling_map}
        self.add_instruction(CZGate(), cz_props)

    def __get_qubit_properties__(self):
        qubit_properties = None
        if ApiAdapter.instance() != None:
            benchmark = ApiAdapter.get_benchmark("monarq")
            for i in range(24):
                if qubit_properties is None:
                    qubit_properties = []
                qubit_properties.append(
                    QubitProperties(
                        t1=benchmark["resultsPerDevice"]["qubits"][str(i)]["t1"],
                        t2=benchmark["resultsPerDevice"]["qubits"][str(i)]["t2Echo"],
                    )
                )
        return qubit_properties
