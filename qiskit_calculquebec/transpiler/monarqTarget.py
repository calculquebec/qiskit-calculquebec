from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit import Parameter
from qiskit.circuit.library import XGate, YGate, ZGate, RZGate, TGate, TdgGate, IGate, CZGate, PhaseGate

from .customGates import Rx90, Rxm90, Ry90, Rym90
from qiskit_calculquebec.monarq_data import connectivity

def get_gate_instance(gate):
    gate_class = gate.__class__
    if gate.params:
        symbolic_params = [Parameter(f"θ{i}") for i in range(len(gate.params))]
        gate_instance = gate_class(*symbolic_params)
    else:
        gate_instance = gate_class()
    return gate_instance

def getTarget():
    target = Target()

    custom_gates = [Rx90(), Rxm90(), Ry90(), Rym90()]
    for gate in custom_gates:
        target.add_instruction(gate, {
            (q,): InstructionProperties() for q in connectivity.get("qubits")
        })

    builtin_gates = [
        IGate(), XGate(), YGate(), ZGate(),
        RZGate(0.1), TGate(), TdgGate(), PhaseGate(0.1), CZGate()
    ]

    for gate in builtin_gates:
        num_qubits = gate.num_qubits
        gate_instance=get_gate_instance(gate)
        
        if num_qubits == 1:
            target.add_instruction(gate_instance, {
                (q,): InstructionProperties() for q in connectivity.get("qubits")
            })
        elif num_qubits == 2:
            instr_props = {}
            for edge in connectivity["couplers"].values():
                instr_props[(edge[0],edge[1])] = InstructionProperties()
                instr_props[(edge[1],edge[0])] = InstructionProperties()

            target.add_instruction(gate_instance, instr_props )
            
    return target