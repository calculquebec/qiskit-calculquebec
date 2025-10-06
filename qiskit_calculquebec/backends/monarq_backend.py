import numpy as np
from qiskit import generate_preset_pass_manager
from qiskit.providers import BackendV2 as Backend
from qiskit.providers import Options
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

from qiskit_calculquebec.API.adapter import ApiAdapter
from qiskit_calculquebec.API.client import ApiClient
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

    def __init__(self, client: ApiClient = None):
        super().__init__()

        # Initialize the device target
        self._target = Yukon()
        self.name = self._target.name

        # Set backend options validators (only shots supported here)
        self.options.set_validator("shots", (1, 1000))

        # Initialize API client if provided
        self._client = client
        if self._client is not None:
            ApiAdapter.initialize(self._client)

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
        return Options(shots=1000)

    def _validate_measurements_at_end(self, circuits):
        """
        Ensure all measurements are only at the end of each qubit and
        have exactly one qubit and one classical bit.
        """
        from qiskit.circuit import Measure

        if not isinstance(circuits, (list, tuple)):
            circuits = [circuits]

        for circuit in circuits:
            for qubit in circuit.qubits:
                # Track the index of the last non-measure instruction for this qubit
                last_non_measure_idx = -1
                for idx, instr in enumerate(circuit.data):
                    if qubit in instr[1] and not isinstance(instr[0], Measure):
                        last_non_measure_idx = idx

                # After the last non-measure instruction, only Measure is allowed
                for idx, instr in enumerate(
                    circuit.data[last_non_measure_idx + 1 :],
                    start=last_non_measure_idx + 1,
                ):
                    if qubit in instr[1]:
                        if not isinstance(instr[0], Measure):
                            raise ValueError(
                                f"Non-measure operation after measurement on qubit {qubit} "
                                f"in circuit '{circuit.name}' at instruction {idx}."
                            )
                        if len(instr[1]) != 1 or len(instr[2]) != 1:
                            raise ValueError(
                                f"Measurement at instruction {idx} in circuit '{circuit.name}' "
                                f"must have exactly one qubit and one classical bit."
                            )

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
        self._validate_measurements_at_end(circuits)

        # Remove barriers to simplify transpilation/execution
        circuits = RemoveBarriers()(circuits)

        shots = kwargs.get("shots", getattr(self.options, "shots", 1000))
        if shots > 1000:
            shots = 1000
            Warning("Shots capped at 1000 for MonarQBackend.")

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
