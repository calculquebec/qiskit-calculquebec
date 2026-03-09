from qiskit_calculquebec.backends.targets.anyon_target import AnyonTarget


class Yukon(AnyonTarget):
    """
    Concrete target description for the Yukon 6-qubit quantum device.

    This class specializes
    :class:`qiskit_calculquebec.backends.targets.anyon_target.AnyonTarget`
    for the Yukon processor by defining its hardware topology, available
    qubits, and device name.

    The Yukon target inherits the gate definitions, instruction properties,
    and calibration retrieval mechanisms implemented in
    :class:`AnyonTarget`.

    Notes
    -----
    The Yukon processor is a small device composed of six qubits arranged in
    a linear topology. Each connection between qubits is represented as a
    directed edge in the coupling map to indicate that two-qubit gates can be
    executed in both directions.

    Examples
    --------
    Instantiate the target:

    .. code-block:: python

        target = Yukon()

    Inspect the topology:

    .. code-block:: python

        print(target.name)
        print(list(target.qubits))
        print(target.coupling_map)
    """

    def coupling_map(self):
        """
        Return the Yukon device coupling map.

        The coupling map describes the directed connectivity between
        physical qubits on the device.

        Returns
        -------
        list[tuple[int, int]]
            List of directed qubit connections representing the device
            hardware topology.
        """
        return [
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

    def qubits(self):
        """
        Return the physical qubit indices of the Yukon device.

        Returns
        -------
        range
            Range object representing the six physical qubits of the device.
        """
        return range(6)

    def device_name(self):
        """
        Return the device name.

        This name is used by the parent class to retrieve calibration and
        benchmark information through the API.

        Returns
        -------
        str
            The device name, ``"Yukon"``.
        """
        return "Yukon"

    def __init__(self):
        """
        Initialize the Yukon target.

        This constructor delegates initialization to the parent
        :class:`AnyonTarget`, which builds the instruction set,
        registers gate properties, and retrieves calibration data
        when available.
        """
        super().__init__()
