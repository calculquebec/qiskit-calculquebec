from qiskit.transpiler.target import Target
from qiskit.circuit.library import (
    IGate, XGate, YGate, ZGate, TGate, TdgGate,
    PhaseGate, CZGate, RZGate, SXGate, SXdgGate, Measure
)
from qiskit.circuit import Parameter

# Custom single-qubit rotations (not in standard Qiskit)
from qiskit_calculquebec.custom_gates.ry_90_gate import RY90Gate
from qiskit_calculquebec.custom_gates.ry_m90_gate import RYm90Gate


class Yukon(Target):
    """
    Custom Qiskit Target for the Yukon 6-qubit device.

    Defines:
    - Qubit connectivity (coupling map)
    - Supported single- and two-qubit gates
    - Measurement operations
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Yukon"

        # Define the bidirectional connectivity of the 6 qubits
        # Each tuple represents a directed edge (control, target)
        self.coupling_map = [
            (0, 1), (1, 0),
            (1, 2), (2, 1),
            (2, 3), (3, 2),
            (3, 4), (4, 3),
            (4, 5), (5, 4)
        ]

        # Define qubits (0 to 5)
        qubits = range(6)

        # Parameter for parameterized gates (RZ and Phase)
        phi = Parameter("φ")

        # --- Single-qubit gates ---
        single_qubit_gates = [
            IGate(), XGate(), YGate(), ZGate(),
            TGate(), TdgGate(),
            RZGate(phi), PhaseGate(phi),
            SXGate(), SXdgGate(),
            Measure(),
            RY90Gate(), RYm90Gate()  # Custom gates
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
