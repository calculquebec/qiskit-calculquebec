from qiskit_calculquebec.backends.targets.anyon_target import AnyonTarget


class MonarQ(AnyonTarget):
    """Concrete target description for the MonarQ 24-qubit quantum device.

    This class specializes
    ``qiskit_calculquebec.backends.targets.anyon_target.AnyonTarget``
    for the MonarQ processor by defining its hardware topology,
    qubit indices, and device name.

    The MonarQ target inherits the default gate set, instruction
    registration, and calibration handling logic from ``AnyonTarget``.

    Note:
        The device topology is described as a directed coupling map.
        Each physical connection is represented in both directions
        when bidirectional execution is supported by the backend.

    Example:
        Instantiate the target:

        .. code-block:: python

            target = MonarQ()

        Access device metadata:

        .. code-block:: python

            print(target.name)
            print(list(target.qubits))
            print(target.coupling_map)
    """

    def coupling_map(self):
        """Return the MonarQ device coupling map.

        The coupling map defines the directed connectivity between
        the physical qubits of the MonarQ processor.

        Returns:
            list[tuple[int, int]]: Directed qubit connections describing the
                hardware connectivity graph.
        """
        return [
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

    def qubits(self):
        """Return the physical qubit indices of the MonarQ device.

        Returns:
            range: Range covering the 24 physical qubits of the processor.
        """
        return range(24)

    def device_name(self):
        """Return the device name.

        This name is used by the parent class to retrieve calibration
        and benchmark information from the API.

        Returns:
            str: The device name ``"MonarQ"``.
        """
        return "MonarQ"

    def __init__(self):
        """Initialize the MonarQ target.

        This constructor delegates initialization to ``AnyonTarget``,
        which:

        * Builds the supported gate set
        * Registers instruction properties
        * Loads calibration data when available
        """
        super().__init__()
