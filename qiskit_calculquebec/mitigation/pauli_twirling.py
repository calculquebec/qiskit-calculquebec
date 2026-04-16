"""
Pauli Twirling (PT) for MonarQ
================================

Pauli Twirling inserts random Pauli gate pairs around two-qubit gates
(CNOT, CZ) to convert an arbitrary noise channel into a depolarizing
channel, which is easier to handle with other mitigation techniques.

Can be combined with ZNE (PT + ZNE) for stronger noise reduction.
"""

import numpy as np

try:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2
except ImportError:
    generate_preset_pass_manager = None  # type: ignore[assignment]
    SamplerV2 = None  # type: ignore[assignment,misc]


def _require_mitiq_pt():
    try:
        from mitiq.pt import generate_pauli_twirl_variants

        return generate_pauli_twirl_variants
    except ImportError:
        raise ImportError(
            "mitiq is required for PauliTwirlingMitigation.\n"
            "Install it with: pip install mitiq\n"
            "or: pip install qiskit-calculquebec[mitigation]"
        )


def _require_mitiq_zne():
    try:
        from mitiq import zne

        return zne
    except ImportError:
        raise ImportError(
            "mitiq is required for PT + ZNE.\n"
            "Install it with: pip install mitiq\n"
            "or: pip install qiskit-calculquebec[mitigation]"
        )


class PauliTwirlingMitigation:
    """Pauli Twirling (PT) for MonarQ, with optional ZNE combination.

    Generates ``num_variants`` twirled copies of the circuit, executes
    each one, and averages the results to reduce variance from the
    randomized Pauli insertions.

    Args:
        backend (MonarQBackend): Calcul Québec backend.
        num_variants (int): Number of twirled variants to average. Higher
            values reduce variance but increase total shot count
            (``num_variants × shots`` executions). Default: 10.
        shots (int): Shots per variant. Default: 1024.

    Examples:
        PT alone:

        >>> pt = PauliTwirlingMitigation(backend, num_variants=10)
        >>> result = pt.run(circuit)

        PT combined with ZNE:

        >>> pt = PauliTwirlingMitigation(backend, num_variants=10)
        >>> result = pt.run_with_zne(circuit, scale_factors=[1.0, 2.0, 3.0])

        PT + ZNE with a custom factory:

        >>> from mitiq.zne.inference import RichardsonFactory
        >>> pt = PauliTwirlingMitigation(backend, num_variants=10)
        >>> result = pt.run_with_zne(
        ...     circuit,
        ...     factory=RichardsonFactory([1.0, 2.0, 3.0]),
        ... )
    """

    def __init__(self, backend, num_variants: int = 10, shots: int = 1024):
        self.backend = backend
        self.num_variants = num_variants
        self.shots = shots

    # ─────────────────────────────────────────────────────────────────────

    def _make_base_executor(self, rem=None, qubits=None):
        """Build a single-shot executor: transpile, run, return P(|0…0⟩).

        Args:
            rem (ReadoutMitigation | None): If provided, REM correction is
                applied to the counts.
            qubits (list[int] | None): Physical qubits; required when ``rem``
                is provided.

        Returns:
            callable: Executor function ``circuit -> float``.
        """
        backend = self.backend
        shots = self.shots

        def executor(circuit):
            circ = circuit.copy()
            if circ.num_clbits == 0:
                circ.measure_all()

            pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
            transpiled = pm.run(circ)
            if not isinstance(transpiled, list):
                transpiled = [transpiled]
            sampler = SamplerV2(mode=backend)
            counts = (
                sampler.run(transpiled, shots=shots)
                .result()[0]
                .join_data()
                .get_counts()
            )
            # Normalize multi-register keys (e.g. "0 0" → "00")
            counts = {"".join(k.split()): v for k, v in counts.items()}
            n = circuit.num_qubits

            if rem is not None and qubits is None:
                raise ValueError("qubits is required when rem is provided.")

            if rem is not None:
                if rem.method == "matrix":
                    counts = rem.apply_correction(counts, qubits=qubits)
                    return counts.get("0" * n, 0) / sum(counts.values())
                else:
                    quasi = rem.apply_correction(counts, qubits=qubits)
                    probs = quasi.nearest_probability_distribution()
                    return probs.get("0" * n, 0.0)

            return counts.get("0" * n, 0) / shots

        return executor

    def _make_pt_executor(self, rem=None, qubits=None):
        """Build a twirling executor that averages over ``num_variants`` variants.

        Generates ``num_variants`` twirled variants, executes each one, and
        returns the mean P(|0…0⟩). Used as the executor inside ZNE
        (``run_with_zne``).

        Args:
            rem (ReadoutMitigation | None): If provided, REM correction is
                applied to each variant.
            qubits (list[int] | None): Physical qubits; required when ``rem``
                is provided.

        Returns:
            callable: Executor function ``circuit -> float``.
        """
        generate_pauli_twirl_variants = _require_mitiq_pt()
        base_executor = self._make_base_executor(rem=rem, qubits=qubits)
        num_variants = self.num_variants

        def pt_executor(circuit):
            variants = generate_pauli_twirl_variants(circuit, num_circuits=num_variants)
            values = [base_executor(v) for v in variants]
            return float(np.mean(values))

        return pt_executor

    def run(self, circuit, rem=None, qubits=None) -> float:
        """Run the circuit with Pauli Twirling and return the averaged result.

        Args:
            circuit (QuantumCircuit): Circuit to execute.
            rem (ReadoutMitigation | None): Optional REM correction applied to
                each variant.
            qubits (list[int] | None): Physical qubits; required when ``rem``
                is provided.

        Returns:
            float: P(|0…0⟩) averaged over ``num_variants`` twirled variants.
        """
        return self._make_pt_executor(rem=rem, qubits=qubits)(circuit)

    def run_unmitigated(self, circuit, rem=None, qubits=None) -> float:
        """Run the circuit without twirling for baseline comparison.

        Args:
            circuit (QuantumCircuit): Circuit to execute.
            rem (ReadoutMitigation | None): Optional REM correction applied to
                the counts.
            qubits (list[int] | None): Physical qubits; required when ``rem``
                is provided.

        Returns:
            float: Raw P(|0…0⟩) without any twirling.
        """
        return self._make_base_executor(rem=rem, qubits=qubits)(circuit)

    def run_with_zne(
        self,
        circuit,
        scale_factors: list[float] | None = None,
        factory=None,
        scale_noise=None,
        rem=None,
        qubits=None,
    ) -> float:
        """Run the circuit with PT + ZNE (twirling combined with zero-noise extrapolation).

        Each point on the ZNE noise curve is computed by a PT executor that
        averages ``num_variants`` twirled variants.

        Args:
            circuit (QuantumCircuit): Circuit to execute. Measurements are
                stripped internally.
            scale_factors (list[float] | None): Noise scale factors. Ignored if
                ``factory`` is provided.
                Default: ``[1.0, 1.5, 2.0, 2.5, 3.0]``.
            factory (mitiq.zne.inference.Factory | None): Extrapolation method.
                ``None`` → ``LinearFactory(scale_factors)``. LinearFactory is
                more stable than Richardson with 4+ scale factors.
            scale_noise (callable | None): Noise scaling method.
                ``None`` → ``fold_gates_at_random``.
            rem (ReadoutMitigation | None): Optional REM correction applied
                inside each variant executor.
            qubits (list[int] | None): Physical qubits; required when ``rem``
                is provided.

        Returns:
            float: Zero-noise extrapolated value from PT + ZNE.
        """
        zne = _require_mitiq_zne()
        pt_executor = self._make_pt_executor(rem=rem, qubits=qubits)

        _scale_factors = scale_factors or [1.0, 1.5, 2.0, 2.5, 3.0]

        kwargs = {}
        if factory is not None:
            kwargs["factory"] = factory
        else:
            kwargs["factory"] = zne.inference.LinearFactory(_scale_factors)
        if scale_noise is not None:
            kwargs["scale_noise"] = scale_noise

        # Strip measurements — mitiq manages them internally via noise folding
        circuit = circuit.remove_final_measurements(inplace=False)

        result = zne.execute_with_zne(circuit, pt_executor, **kwargs)
        return float(result.real) if hasattr(result, "real") else float(result)

    def run_variants(self, circuit, rem=None, qubits=None) -> list[float]:
        """Return the individual result of each twirled variant without averaging.

        Useful for inspecting the variance introduced by twirling.

        Args:
            circuit (QuantumCircuit): Circuit to execute.
            rem (ReadoutMitigation | None): Optional REM correction applied to
                each variant.
            qubits (list[int] | None): Physical qubits; required when ``rem``
                is provided.

        Returns:
            list[float]: P(|0…0⟩) for each of the ``num_variants`` twirled
                variants.
        """
        generate_pauli_twirl_variants = _require_mitiq_pt()
        base_executor = self._make_base_executor(rem=rem, qubits=qubits)
        variants = generate_pauli_twirl_variants(
            circuit, num_circuits=self.num_variants
        )
        return [base_executor(v) for v in variants]
