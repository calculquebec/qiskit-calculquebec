from qiskit_calculquebec.backends.targets.anyon_target import AnyonTarget


class Yukon(AnyonTarget):
    """
    Custom Qiskit Target for the Yukon 6-qubit device.

    Defines:
    - Qubit connectivity (coupling map)
    - Supported single- and two-qubit gates
    - Measurement operations
    """

    def __init__(self):
        self.qubits = range(6)
        self.coupling_map = [
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
        self.name = "Yukon"
        super().__init__(self.coupling_map, self.qubits, self.name)
