from qiskit.circuit import Parameter
from qiskit.circuit.library import RXGate, RYGate, U3Gate, CXGate
from qiskit.circuit.equivalence_library import EquivalenceLibrary
from qiskit.circuit import QuantumCircuit
from numpy import pi
from .customGates import Rx90, Rxm90, Ry90, Rym90


def getEquivalenceRules():

    custom_lib = EquivalenceLibrary()

    theta = Parameter("θ")
    phi = Parameter("φ")
    lam = Parameter("λ")

    # ZYZ decomposition: U3(θ, φ, λ) ≈ RZ(φ) → RY(θ) → RZ(λ)
    u3_equiv = QuantumCircuit(1)
    u3_equiv.rz(phi, 0)
    u3_equiv.append(Rxm90(), [0])
    u3_equiv.rz(theta, 0)
    u3_equiv.append(Rx90(), [0])
    u3_equiv.rz(lam, 0)

    custom_lib.add_equivalence(U3Gate(theta, phi, lam), u3_equiv)

    # CX ≈ RY(-π/2) on target → CZ → RY(π/2) on target
    cx_equiv = QuantumCircuit(2)
    cx_equiv.append(Rym90(), [1])
    cx_equiv.cz(0, 1)
    cx_equiv.append(Ry90(), [1])

    custom_lib.add_equivalence(CXGate(), cx_equiv)

    # RX(θ) ≈ RY(π/2) → RZ(θ) → RY(-π/2)
    rx_equiv = QuantumCircuit(1)
    rx_equiv.append(Rym90(), [0])
    rx_equiv.rz(-theta, 0)
    rx_equiv.append(Ry90(), [0])

    custom_lib.add_equivalence(RXGate(theta), rx_equiv)

    # RY(θ) ≈ RX(π/2) → RZ(θ) → RX(-π/2)
    ry_equiv = QuantumCircuit(1)
    ry_equiv.append(Rx90(), [0])
    ry_equiv.rz(theta, 0)
    ry_equiv.append(Rxm90(), [0])

    custom_lib.add_equivalence(RYGate(theta), ry_equiv)

    return custom_lib
