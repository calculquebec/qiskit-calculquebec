from qiskit_calculquebec.backends.targets.anyon_target import AnyonTarget


class MonarQ(AnyonTarget):
    """
    Custom Qiskit Target for the MonarQ 24-qubit device.

    Defines:
    - Qubit connectivity (coupling map)
    - Supported single- and two-qubit gates
    - Measurement operations
    """

    def __init__(self):
        self.qubits = range(24)
        self.coupling_map = [
            (0, 4),
            (4, 0),
            (1, 4),
            (4, 1),
            (1, 5),
            (5, 1),
            (2, 5),
            (5, 2),
            (2, 6),
            (6, 2),
            (3, 6),
            (6, 3),
            (3, 7),
            (7, 3),
            (4, 8),
            (8, 4),
            (4, 9),
            (9, 4),
            (5, 9),
            (9, 5),
            (5, 10),
            (10, 5),
            (6, 10),
            (10, 6),
            (6, 11),
            (11, 6),
            (7, 11),
            (11, 7),
            (8, 12),
            (12, 8),
            (9, 12),
            (12, 9),
            (9, 13),
            (13, 9),
            (10, 13),
            (13, 10),
            (10, 14),
            (14, 10),
            (11, 14),
            (14, 11),
            (11, 15),
            (15, 11),
            (12, 16),
            (16, 12),
            (12, 17),
            (17, 12),
            (13, 17),
            (17, 13),
            (13, 18),
            (18, 13),
            (14, 18),
            (18, 14),
            (14, 19),
            (19, 14),
            (15, 19),
            (19, 15),
            (16, 20),
            (20, 16),
            (17, 20),
            (20, 17),
            (17, 21),
            (21, 17),
            (18, 21),
            (21, 18),
            (18, 22),
            (22, 18),
            (19, 22),
            (22, 19),
            (19, 23),
            (23, 19),
        ]

        self.name = "MonarQ"
        super().__init__(self.coupling_map, self.qubits, self.name)
