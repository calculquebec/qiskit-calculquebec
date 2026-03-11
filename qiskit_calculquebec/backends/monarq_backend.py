import warnings
import math
import numpy as np
from qiskit.circuit import Measure, Delay
from qiskit.circuit.library import IGate
from qiskit import generate_preset_pass_manager
from qiskit.providers import BackendV2 as Backend
from qiskit.providers import Options
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.dagcircuit import DAGOpNode

from qiskit_calculquebec.API.adapter import ApiAdapter
from qiskit_calculquebec.API.client import ApiClient
from qiskit_calculquebec.backends.targets.anyon_target import DT
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
    def dt(self) -> float:
        """Return the system time resolution of input signals in seconds.

        This value (``dt``) is the hardware clock cycle used by the Qiskit
        transpiler to convert gate and :class:`~qiskit.circuit.Delay` durations
        expressed in seconds into integer multiples of the hardware timestep.

        Returns
        -------
        float
            Clock period in seconds (32 ns for Anyon devices).
        """
        return DT

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

        if str.lower(machine_name) not in ["yukon", "monarq"]:
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
            warnings.warn(
                "MonarQBackend supports a maximum of 1024 shots. "
                "Your requested number of shots has been set to 1024.",
                UserWarning,
            )

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

    class DelayToIdentityPass(TransformationPass):
        """
        Transpiler pass to expand each :class:`~qiskit.circuit.Delay` into a
        sequence of :class:`~qiskit.circuit.library.IGate` operations.

        On Anyon hardware, ``IGate`` is the native idle instruction and
        corresponds to exactly one hardware clock cycle (``dt = 32 ns``).
        A ``Delay`` of *n* dt is therefore equivalent to *n* consecutive
        ``IGate`` applications on the same qubit.

        The duration of the ``Delay`` must be expressed in ``dt`` units
        (i.e. the circuit must have been scheduled before this pass runs,
        or the delay must have been inserted with ``unit='dt'``).  Delays
        given in seconds are converted to dt by dividing by ``DT`` and
        rounding to the nearest integer; a warning is emitted when rounding
        is necessary.

        Parameters
        ----------
        dt : float
            Hardware clock period in seconds.  Defaults to
            :data:`~qiskit_calculquebec.backends.targets.anyon_target.DT`.

        Examples
        --------
        >>> pass_ = MonarQBackend.DelayToIdentityPass()
        >>> pm = PassManager([pass_])
        >>> expanded = pm.run(scheduled_circuit)
        """

        def __init__(self, dt: float = DT):
            super().__init__()
            self.dt = dt

        def run(self, dag):
            for node in dag.op_nodes():
                if not isinstance(node.op, Delay):
                    continue

                duration = node.op.duration
                unit = node.op.unit

                # --- resolve duration to an integer number of dt cycles ---
                if unit == "dt":
                    n_cycles = int(duration)
                else:
                    # convert seconds (or ns/µs) to seconds first
                    unit_to_seconds = {"s": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
                    duration_s = duration * unit_to_seconds.get(unit, 1)
                    n_cycles_exact = duration_s / self.dt
                    n_cycles = round(n_cycles_exact)
                    if not math.isclose(n_cycles_exact, n_cycles, rel_tol=1e-6):
                        warnings.warn(
                            f"Delay duration {duration} [{unit}] is not an exact "
                            f"multiple of dt={self.dt} s. "
                            f"Rounded from {n_cycles_exact:.4f} to {n_cycles} IGate(s).",
                            UserWarning,
                        )

                if n_cycles <= 0:
                    dag.remove_op_node(node)
                    continue

                # --- replace the Delay node with n_cycles IGate nodes ---
                qubit = node.qargs[0]
                dag.remove_op_node(node)
                # re-fetch the predecessor/successor so we can insert in-place;
                # substitute_node is unavailable after remove, so we use
                # apply_operation_back on the original DAG — the relative order
                # is preserved because all other nodes are untouched.
                for _ in range(n_cycles):
                    dag.apply_operation_back(IGate(), qargs=(qubit,), cargs=())

            return dag

    def transpile(self, circuit):
        """
        Transpile a circuit with:
        1. Replacement of RY(±π/2) gates
        2. Expansion of Delay gates into sequences of IGate
        3. Level-3 preset optimization passes
        """
        # Step 1: create Qiskit level-3 preset pass manager
        pm3 = generate_preset_pass_manager(optimization_level=3, backend=self)

        # Step 2: create pass managers for RY replacement and Delay expansion
        pm_replace = PassManager([self.ReplaceRYPass()])
        pm_delay = PassManager([self.DelayToIdentityPass(dt=DT)])

        # Step 3: replacement → delay expansion → level-3 passes
        def run_custom_pm(circuit_or_dag):
            dag = circuit_or_dag
            dag = pm_replace.run(dag)
            dag = pm_delay.run(dag)
            dag = pm3.run(dag)
            return dag

        return run_custom_pm(circuit)