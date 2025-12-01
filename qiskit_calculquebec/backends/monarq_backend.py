import warnings
import numpy as np
from qiskit.circuit import Measure
from qiskit import generate_preset_pass_manager
from qiskit.providers import BackendV2 as Backend
from qiskit.providers import Options
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

from qiskit_calculquebec.API.adapter import ApiAdapter
from qiskit_calculquebec.API.client import ApiClient
from qiskit_calculquebec.backends.targets.monarq import MonarQ
from qiskit_calculquebec.backends.targets.yukon import Yukon
from qiskit_calculquebec.backends.utils.job import MultiMonarQJob
from qiskit_calculquebec.custom_gates.ry_90_gate import RY90Gate
from qiskit_calculquebec.custom_gates.ry_m90_gate import RYm90Gate


class MonarQBackend(Backend):
    """
    Custom backend for the Yukon 6-qubit device.

    Features:
    - Integrates with Calcul Québec API
    - Validates measurement placement
    - Supports multi-circuit jobs with MultiMonarQJob
    - Transpilation includes automatic RY(±π/2) replacement
    """

    _client: ApiClient

    @property
    def target(self):
        """Return the backend target object."""
        return self._target

    @property
    def max_circuits(self):
        """Return the maximum number of circuits that can be submitted at once."""
        return 1000

    @classmethod
    def _default_options(cls):
        """Provide default backend options."""
        return Options(shots=1024)

    def __init__(self, machine_name: str = "monarq", client: ApiClient = None):
        super().__init__()
        if client is None:
            raise ValueError("An ApiClient instance must be provided.")

        self._client = client
        ApiAdapter.initialize(self._client)

        if str.lower(self.machine_name) not in ["yukon", "monarq"]:
            raise ValueError(
                f"Unsupported machine name: {self._client.machine_name} please choose 'yukon' or 'monarq'."
            )
        if machine_name.lower() == "yukon":
            self._client.machine_name = "yukon"
            self._target = Yukon()
        elif machine_name.lower() == "monarq":
            self._client.machine_name = "monarq"
            self._target = MonarQ()

        self.name = self._target.name

        # Set backend options validators (only shots supported here)
        self.options.set_validator("shots", (1, 1024))

    def _validate_circuit(self, circuits):
        for qc in circuits:
            measured_qubits = set()
            for instr in qc.data:
                op = instr.operation
                qubits = instr.qubits

                # Mark qubits as measured
                if isinstance(op, Measure):
                    if len(instr.qubits) != 1 or len(instr.clbits) != 1:
                        raise ValueError("Multi-qubit measurements are not supported.")
                    measured_qubits.update(qubits)

                else:
                    # Gate after measurement → raise error
                    if any(q in measured_qubits for q in qubits):
                        raise ValueError(
                            "Gate applied after measurement is not allowed."
                        )
            if len(measured_qubits) == 0:
                raise ValueError("All circuits must contain at least one measurement.")

    def run(self, circuits, **kwargs):
        """
        Submit circuits to the backend and return a MultiMonarQJob.

        Args:
            circuits: Circuit or list of circuits to execute.
            shots: Optional number of shots to execute (capped at 1000).
        """
        if not isinstance(circuits, (list, tuple)):
            circuits = [circuits]

        # Validate measurement placement
        self._validate_circuit(circuits)

        # Remove barriers to simplify transpilation/execution
        circuits = RemoveBarriers()(circuits)

        shots = kwargs.get("shots", getattr(self.options, "shots", 1024))
        if shots > 1024:
            shots = 1024
            warnings.warn("Shots are set at 1024 for MonarQBackend.", UserWarning)

        # Return a multi-job wrapper to handle sequential execution
        return MultiMonarQJob(self, circuits, shots=shots)

    class ReplaceRYPass(TransformationPass):
        """
        Transpiler pass to replace RY(±π/2) with custom RY90/RYm90 gates.
        """

        def run(self, dag):
            for node in dag.op_nodes():
                if node.name == "ry" and np.isclose(node.op.params[0], np.pi / 2):
                    dag.substitute_node(node, RY90Gate())
                elif node.name == "ry" and np.isclose(node.op.params[0], -np.pi / 2):
                    dag.substitute_node(node, RYm90Gate())
            return dag

    def transpile(self, circuit):
        """
        Transpile a circuit with:
        1. Replacement of RY(±π/2) gates
        2. Level-3 preset optimization passes
        """
        # Step 1: create Qiskit level-3 preset pass manager
        pm3 = generate_preset_pass_manager(optimization_level=3, backend=self)

        # Step 2: create a small pass manager for RY replacement
        pm_replace = PassManager([self.ReplaceRYPass()])

        # Step 3: run replacement first, then level-3 passes
        def run_custom_pm(circuit_or_dag):
            dag = circuit_or_dag
            dag = pm_replace.run(dag)
            dag = pm3.run(dag)
            return dag

        return run_custom_pm(circuit)
