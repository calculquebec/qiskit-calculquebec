from qiskit.circuit import Gate, QuantumCircuit
import numpy as np


class Rx90(Gate):
    def __init__(self):
        super().__init__("rx90", 1, [])

    def _define(self):
        qc = QuantumCircuit(1, name=self.name)
        qc.rx(np.pi / 2, 0)
        self.definition = qc


class Rxm90(Gate):
    def __init__(self):
        super().__init__("rxm90", 1, [])

    def _define(self):
        qc = QuantumCircuit(1, name=self.name)
        qc.rx(-np.pi / 2, 0)
        self.definition = qc


class Ry90(Gate):
    def __init__(self):
        super().__init__("ry90", 1, [])

    def _define(self):
        qc = QuantumCircuit(1, name=self.name)
        qc.ry(np.pi / 2, 0)
        self.definition = qc


class Rym90(Gate):
    def __init__(self):
        super().__init__("rym90", 1, [])

    def _define(self):
        qc = QuantumCircuit(1, name=self.name)
        qc.ry(-np.pi / 2, 0)
        self.definition = qc
