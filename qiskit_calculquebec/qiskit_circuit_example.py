from monarq_backend import monarq_backend
from qiskit import QuantumCircuit

backend = monarq_backend()

qc = QuantumCircuit(1, 1)
qc.x(0)
# qc.cx(0, 1)
qc.measure([0], [0])


host1 = "https://manager.anyonlabs.com/"
# host2= "https://monarq.calculquebec.ca"
print(
    backend.run(
        qc,
        circuit_name="test_circuit_name",
        shots=1024,
        host=host1,
        user="username",
        access_token="token",
        project_name="project",
    )
)
