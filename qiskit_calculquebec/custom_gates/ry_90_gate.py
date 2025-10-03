import math
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary


class RY90Gate(Gate):
    """
    Custom single-qubit gate representing RY(π/2).

    Useful for backends with a native RY90 gate or for simplifying transpilation.
    """

    def __init__(self):
        # Initialize gate with name 'ry90', acting on 1 qubit, no parameters
        super().__init__('ry90', 1, [])

    def _define(self):
        """
        Decompose this gate in terms of standard Qiskit gates.
        Here, it is a single RY rotation by π/2.
        """
        qc = QuantumCircuit(1, name=self.name)
        qc.ry(math.pi / 2, 0)  # Apply RY(π/2) on qubit 0
        self.definition = qc


# Register equivalence in Qiskit's session equivalence library
# so the transpiler recognizes RY90Gate as equivalent to RY(π/2)
qc_standard = QuantumCircuit(1)
qc_standard.ry(math.pi / 2, 0)
SessionEquivalenceLibrary.add_equivalence(RY90Gate(), qc_standard)
