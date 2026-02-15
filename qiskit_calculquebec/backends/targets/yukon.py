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
from qiskit.transpiler import InstructionProperties

# Custom single-qubit rotations (not in standard Qiskit)
from qiskit_calculquebec.API.adapter import ApiAdapter
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
        # Define qubits (0 to 5)
        self.qubits = range(6)
        # Define the bidirectional connectivity of the 6 qubits
        # Each tuple represents a directed edge (control, target)
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
        qubit_properties, gate_properties = self.__get_qubit_properties__()
        self.qubit_properties = qubit_properties
        self.name = "Yukon"

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
            if type(gate) == Measure():
                gate_props = {
                    (q,): InstructionProperties(
                        duration=5e-6, error=gate_properties["measure"][q]
                    )
                    for q in self.qubits
                }
            else:
                gate_props = {
                    (q,): InstructionProperties(
                        duration=5e-8, error=gate_properties["single"][q]
                    )
                    for q in self.qubits
                }
            self.add_instruction(gate, gate_props)

        # --- Two-qubit gates ---
        # Only CZ is supported, defined for all edges in the coupling map
        cz_props = {
            edge: InstructionProperties(
                duration=1e-7, error=gate_properties["double"][q]
            )
            for q, edge in enumerate(self.coupling_map)
        }
        self.add_instruction(CZGate(), cz_props)

    def __get_qubit_properties__(self):
        gate_properties = {
            "single": {},
            "measure": {},
            "double": {},
        }

        # Sensible defaults if benchmark is unavailable
        default_single_err = 1e-3
        default_meas_err = 2e-2
        default_cz_err = 2e-2

        qubit_properties = [QubitProperties(t1=None, t2=None) for _ in self.qubits]

        adapter = ApiAdapter.instance()
        if adapter is not None:
            benchmark = ApiAdapter.get_benchmark("yukon")

            for i in self.qubits:
                qb = benchmark["resultsPerDevice"]["qubits"][str(i)]
                qubit_properties[i] = QubitProperties(
                    t1=qb.get("t1", None),
                    t2=qb.get("t2Echo", None),
                )

                gate_properties["single"][i] = 1 - qb.get(
                    "parallelSingleQubitGateFidelity", 1 - default_single_err
                )
                gate_properties["measure"][i] = 1 - qb.get(
                    "parallelReadoutState1Fidelity", 1 - default_meas_err
                )

            couplers = benchmark["resultsPerDevice"].get("couplers", {})
            for idx in range(len(self.coupling_map)):
                c = couplers.get(str(idx), {})
                gate_properties["double"][idx] = 1 - c.get(
                    "czGateFidelity", 1 - default_cz_err
                )

        else:
            # No API: fill defaults
            for i in self.qubits:
                gate_properties["single"][i] = default_single_err
                gate_properties["measure"][i] = default_meas_err
            for idx in range(len(self.coupling_map)):
                gate_properties["double"][idx] = default_cz_err

        return qubit_properties, gate_properties
