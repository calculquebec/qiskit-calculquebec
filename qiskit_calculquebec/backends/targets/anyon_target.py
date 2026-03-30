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
from qiskit.circuit import Parameter, Delay
from qiskit.transpiler.target import QubitProperties
from qiskit.transpiler import InstructionProperties

from qiskit_calculquebec.API.adapter import ApiAdapter
from qiskit_calculquebec.custom_gates.ry_90_gate import RY90Gate
from qiskit_calculquebec.custom_gates.ry_m90_gate import RYm90Gate

from abc import ABC, abstractmethod


DT = 32e-9


class AnyonTarget(Target, ABC):
    """
    Abstract base class describing a quantum hardware target for Anyon devices.

    This class extends :class:`qiskit.transpiler.Target` and provides a common
    interface for defining the characteristics of Anyon-based quantum devices,
    such as Yukon or MonarQ.

    The target defines:

    * The physical qubits available on the device
    * The device coupling map
    * The supported gate set
    * Gate durations and error rates
    * Measurement operations

    Hardware calibration data (gate errors, measurement errors, and coherence
    times) are optionally retrieved through the
    :class:`qiskit_calculquebec.API.adapter.ApiAdapter`.

    Subclasses must implement methods describing the hardware topology.

    Notes
    -----
    This class is abstract and cannot be instantiated directly. Concrete
    subclasses must implement:

    * :meth:`coupling_map`
    * :meth:`qubits`
    * :meth:`device_name`

    Examples
    --------

    Example of a concrete device target:

    .. code-block:: python

        class Yukon(AnyonTarget):

            def coupling_map(self):
                return [(0,1),(1,0),(1,2),(2,1)]

            def qubits(self):
                return list(range(6))

            def device_name(self):
                return "Yukon"
    """

    @abstractmethod
    def coupling_map(self):
        """
        Return the device coupling map.

        The coupling map defines the connectivity between physical qubits.

        Returns
        -------
        list[tuple[int, int]]
            List of directed qubit connections.
        """
        pass

    @abstractmethod
    def qubits(self):
        """
        Return the list of physical qubits available on the device.

        Returns
        -------
        list[int]
            Indices of physical qubits.
        """
        pass

    @abstractmethod
    def device_name(self):
        """
        Return the name of the quantum device.

        Returns
        -------
        str
            Device name used to retrieve calibration data.
        """
        pass

    def __init__(self):
        """
        Initialize the hardware target.

        This constructor:

        * Initializes the Qiskit :class:`Target`
        * Loads qubit properties
        * Defines the default gate set
        * Registers supported instructions with their associated
          duration and error rates.
        """
        super().__init__()
        self.dt = DT

        self.qubits = self.qubits()
        self.coupling_map = self.coupling_map()
        self.name = self.device_name()

        qubit_properties, gate_properties = self.__get_qubit_properties__()
        self.qubit_properties = qubit_properties

        phi = Parameter("φ")

        self.__define_default_gates__(phi)
        self.__set_single_qubit_gate_properties__(gate_properties)
        self.__set_two_qubit_gate_properties__(gate_properties)

    def __set_two_qubit_gate_properties__(self, gate_properties):
        """
        Register two-qubit gates supported by the device.

        Currently only the :class:`CZGate` is supported.

        Parameters
        ----------
        gate_properties : dict
            Dictionary containing calibrated gate error rates.
        """
        cz_props = {
            edge: InstructionProperties(
                duration=1e-7,
                error=gate_properties["double"][q // 2],
            )
            for q, edge in enumerate(self.coupling_map)
        }

        self.add_instruction(CZGate(), cz_props)

    def __set_single_qubit_gate_properties__(self, gate_properties):
        """
        Register single-qubit gates for all qubits.

        Each instruction is added to the target with associated duration
        and error properties.

        Parameters
        ----------
        gate_properties : dict
            Dictionary containing single-qubit and measurement errors.
        """
        for gate in self.default_single_qubit_gates:

            if isinstance(gate, Measure):

                gate_props = {
                    (q,): InstructionProperties(
                        duration=4e-7,
                        error=gate_properties["measure"][q],
                    )
                    for q in self.qubits
                }

            elif isinstance(
                gate,
                (
                    RZGate,
                    ZGate,
                    TGate,
                    TdgGate,
                    PhaseGate,
                ),
            ):
                gate_props = {
                    (q,): InstructionProperties(duration=0, error=0)
                    for q in self.qubits
                }

            elif isinstance(gate, Delay):
                gate_props = {
                    (q,): InstructionProperties(duration=None, error=0)
                    for q in self.qubits
                }

            else:

                gate_props = {
                    (q,): InstructionProperties(
                        duration=5e-8,
                        error=gate_properties["single"][q],
                    )
                    for q in self.qubits
                }

            self.add_instruction(gate, gate_props)

    def __define_default_gates__(self, phi):
        """
        Define the default gate set supported by Anyon devices.

        Parameters
        ----------
        phi : Parameter
            Parameter used for parameterized rotation gates.
        """

        self.default_single_qubit_gates = [
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
            RYm90Gate(),
            Delay(phi),
        ]

        self.default_two_qubit_gates = [CZGate()]

    def __get_qubit_properties__(self):
        """
        Retrieve device calibration data.

        Calibration data is retrieved from the Anyon API when available.
        If the API is unavailable, default error values are used.

        Returns
        -------
        tuple[list[QubitProperties], dict]
            * List of qubit properties (T1, T2)
            * Dictionary containing gate error information
        """

        gate_properties = {
            "single": {},
            "measure": {},
            "double": {},
        }

        default_single_err = 1e-3
        default_meas_err = 2e-2
        default_cz_err = 2e-2

        qubit_properties = [QubitProperties(t1=None, t2=None) for _ in self.qubits]

        adapter = ApiAdapter.instance()

        if adapter is not None:

            benchmark = ApiAdapter.get_benchmark(self.name.lower())

            for i in self.qubits:

                qb = benchmark["resultsPerDevice"]["qubits"][str(i)]

                qubit_properties[i] = QubitProperties(
                    t1=qb.get("t1", None),
                    t2=qb.get("t2Echo", None),
                )

                gate_properties["single"][i] = 1 - qb.get(
                    "parallelSingleQubitGateFidelity",
                    1 - default_single_err,
                )

                gate_properties["measure"][i] = 1 - (
                    (
                        qb.get(
                            "parallelReadoutState1Fidelity",
                            1 - default_meas_err,
                        )
                        + qb.get(
                            "parallelReadoutState0Fidelity",
                            1 - default_meas_err,
                        )
                    )
                    / 2
                )

            couplers = benchmark["resultsPerDevice"].get("couplers", {})
            for idx in range(len(self.coupling_map)):
                c = couplers.get(str(idx), {})
                gate_properties["double"][idx] = 1 - c.get(
                    "czGateFidelity",
                    1 - default_cz_err,
                )

        else:

            for i in self.qubits:
                gate_properties["single"][i] = default_single_err
                gate_properties["measure"][i] = default_meas_err

            for idx in range(len(self.coupling_map)):
                gate_properties["double"][idx] = default_cz_err

        return qubit_properties, gate_properties
