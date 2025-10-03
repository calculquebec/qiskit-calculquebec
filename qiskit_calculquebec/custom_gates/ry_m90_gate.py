import math
from qiskit.circuit.library import RYGate
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary


class RYm90Gate(Gate):
    """
    Custom single-qubit gate representing RY(-π/2).

    This is useful for backends that have a native RY90-like gate,
    or for simplifying transpilation.
    """

    def __init__(self):
        # Initialize gate with name 'rym90', acting on 1 qubit, with no parameters
        super().__init__('rym90', 1, [])

    def _define(self):
        """
        Define the decomposition of this gate in terms of standard Qiskit gates.
        Here, it is just an RY rotation by -π/2.
        """
        qc = QuantumCircuit(1, name=self.name)
        qc.ry(-math.pi / 2, 0)  # Apply RY(-π/2) on qubit 0
        self.definition = qc


# Register the equivalence in Qiskit's session equivalence library
# so that transpiler knows RYm90Gate == RY(-π/2)
qc_rym90 = QuantumCircuit(1)
qc_rym90.append(RYGate(-math.pi / 2), [0])
SessionEquivalenceLibrary.add_equivalence(RYm90Gate(), qc_rym90)
