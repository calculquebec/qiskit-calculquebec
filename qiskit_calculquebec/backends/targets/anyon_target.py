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


class AnyonTarget(Target):
    """
    Custom Qiskit Target for the Yukon 6-qubit device.

    Defines:
    - Qubit connectivity (coupling map)
    - Supported single- and two-qubit gates
    - Measurement operations
    """

    def __init__(self, coupling_map, qubits, name: str):
        super().__init__()
        self.qubits = qubits
        self.coupling_map = coupling_map
        self.name = name
        qubit_properties, gate_properties = self.__get_qubit_properties__()
        self.qubit_properties = qubit_properties

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
            if isinstance(gate, Measure):
                gate_props = {
                    (q,): InstructionProperties(
                        duration=4e-7, error=gate_properties["measure"][q]
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
                duration=1e-7, error=gate_properties["double"][q // 2]
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
            benchmark = ApiAdapter.get_benchmark(self.name.lower())

            for i in self.qubits:
                qb = benchmark["resultsPerDevice"]["qubits"][str(i)]
                qubit_properties[i] = QubitProperties(
                    t1=qb.get("t1", None),
                    t2=qb.get("t2Echo", None),
                )

                gate_properties["single"][i] = 1 - qb.get(
                    "parallelSingleQubitGateFidelity", 1 - default_single_err
                )
                gate_properties["measure"][i] = 1 - (
                    (
                        qb.get("parallelReadoutState1Fidelity", 1 - default_meas_err)
                        + qb.get("parallelReadoutState0Fidelity", 1 - default_meas_err)
                    )
                    / 2
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
