from qiskit import QuantumCircuit
from copy import deepcopy

def remove_measurements(circuit: QuantumCircuit):
    """Retourne un circuit sans mesures et une map des mesures originales."""
    new_circ = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    new_circ.metadata = deepcopy(circuit.metadata)

    measures = []

    for inst, qargs, cargs in circuit.data:
        if inst.name == "measure":
            q_index = circuit.qubits.index(qargs[0])
            c_index = circuit.clbits.index(cargs[0])
            measures.append((q_index, c_index))
        else:
            new_circ.append(inst, qargs, cargs)

    return new_circ, measures


def append_measurements(circuit: QuantumCircuit, measure_map):
    """Combine un circuit sans mesures et une map des mesures."""
    for q_idx, c_idx in measure_map:
        circuit.measure(q_idx, c_idx)
    return circuit