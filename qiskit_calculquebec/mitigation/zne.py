"""
Zero-Noise Extrapolation (ZNE) for MonarQ
==========================================

Wraps the MonarQ executor and delegates extrapolation to mitiq.zne.
``optimization_level=0`` is mandatory for ZNE: the pass manager must not
modify the circuit after noise folding applied by mitiq.
"""


try:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2
except ImportError:
    generate_preset_pass_manager = None  # type: ignore[assignment]
    SamplerV2 = None  # type: ignore[assignment,misc]


def _require_mitiq_zne():
    try:
        from mitiq import zne
        return zne
    except ImportError:
        raise ImportError(
            "mitiq is required for ZNEMitigation.\n"
            "Install it with: pip install mitiq\n"
            "or: pip install qiskit-calculquebec[mitigation]"
        )


class ZNEMitigation:
    """Zero-Noise Extrapolation (ZNE) for MonarQ.

    Runs the circuit at several noise scale factors, then extrapolates
    to the zero-noise limit using the chosen inference factory.

    Args:
        backend (MonarQBackend): Calcul Québec backend.
        scale_factors (list[float] | None): Noise scale factors.
            Default: ``[1.0, 1.5, 2.0, 2.5, 3.0]``.
        factory (mitiq.zne.inference.Factory | None): Extrapolation method.
            ``None`` → ``LinearFactory(scale_factors)``. Richardson is
            theoretically more accurate but tends to diverge with 4+ scale
            factors. Examples: ``LinearFactory([1,2,3])``,
            ``RichardsonFactory([1,2,3])``,
            ``ExpFactory([1,2,3], asymptote=0.5)``.
        scale_noise (callable | None): Noise scaling method.
            ``None`` → ``fold_gates_at_random`` (mitiq default).
            Alternative: ``fold_global``.
        shots (int): Shots per circuit. Default: 1024.

    Examples:
        Default observable — P(|0…0⟩):

        >>> zne_mit = ZNEMitigation(backend, scale_factors=[1.0, 2.0, 3.0])
        >>> result = zne_mit.run(circuit)
        >>> print(f"Raw: {zne_mit.run_unmitigated(circuit):.4f}  Mitigated: {result:.4f}")

        Arbitrary Pauli observable:

        >>> from mitiq import Observable, PauliString
        >>> obs = Observable(PauliString("ZZ", support=[0, 1]))  # <Z0 Z1>
        >>> result = zne_mit.run(circuit, observable=obs)

        Custom factory:

        >>> from mitiq.zne.inference import LinearFactory
        >>> from mitiq.zne.scaling import fold_global
        >>> zne_mit = ZNEMitigation(
        ...     backend,
        ...     factory=LinearFactory([1.0, 1.5, 2.0, 2.5, 3.0]),
        ...     scale_noise=fold_global,
        ... )
        >>> result = zne_mit.run(circuit)

        ZNE combined with REM:

        >>> rem = ReadoutMitigation(backend, method='m3')
        >>> rem.cals_from_system()
        >>> pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
        >>> t = pm.run(circuit)
        >>> physical_qubits = (
        ...     [t.layout.final_layout[q] for q in t.qubits]
        ...     if t.layout and t.layout.final_layout
        ...     else list(range(circuit.num_qubits))
        ... )
        >>> result = zne_mit.run(circuit, rem=rem, qubits=physical_qubits)
    """

    def __init__(
        self,
        backend,
        scale_factors: list[float] | None = None,
        factory=None,
        scale_noise=None,
        shots: int = 1024,
    ):
        self.backend = backend
        self.scale_factors = scale_factors or [1.0, 1.5, 2.0, 2.5, 3.0]
        self.factory = factory
        self.scale_noise = scale_noise
        self.shots = shots

    # ─────────────────────────────────────────────────────────────────────

    def _make_executor(self, rem=None, qubits=None, observable=None):
        """Build a mitiq-compatible executor for this backend.

        ``optimization_level=0`` is mandatory: the transpiler must not alter
        the circuit after mitiq's noise folding.

        Two modes depending on ``observable``:

        - ``observable=None``: returns ``float`` — P(|0…0⟩).
        - ``observable`` provided: returns ``MeasurementResult`` (raw
          bitstrings); mitiq computes the expectation value internally.

        Args:
            rem (ReadoutMitigation | None): If provided, REM correction is
                applied inside the executor.
            qubits (list[int] | None): Physical qubits; required when ``rem``
                is provided.
            observable (mitiq.Observable | None): If provided, the executor
                returns ``MeasurementResult`` instead of ``float``.
        """
        backend = self.backend
        shots = self.shots

        if observable is not None:
            import numpy as np
            from mitiq import MeasurementResult

            def executor(circuit) -> MeasurementResult:
                circ = circuit.copy()
                if circ.num_clbits == 0:
                    circ.measure_all()

                pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
                transpiled = pm.run(circ)
                if not isinstance(transpiled, list):
                    transpiled = [transpiled]
                sampler = SamplerV2(mode=backend)
                counts = sampler.run(transpiled, shots=shots).result()[0].join_data().get_counts()
                # Normalize multi-register keys (e.g. "0 0" → "00")
                counts = {"".join(k.split()): v for k, v in counts.items()}

                if rem is not None and qubits is None:
                    raise ValueError("qubits is required when rem is provided.")

                if rem is not None:
                    if rem.method == "matrix":
                        counts = rem.apply_correction(counts, qubits=qubits)
                    else:
                        quasi = rem.apply_correction(counts, qubits=qubits)
                        total = sum(counts.values())
                        counts = {
                            k: max(0, int(round(v * total)))
                            for k, v in quasi.nearest_probability_distribution().items()
                        }

                bitstrings = []
                for bitstring, count in counts.items():
                    bitstrings.extend([[int(b) for b in bitstring]] * count)
                return MeasurementResult(np.array(bitstrings, dtype=int))

        else:
            def executor(circuit):
                # mitiq strips measurements before folding — re-add them if needed
                circ = circuit.copy()
                if circ.num_clbits == 0:
                    circ.measure_all()

                pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
                transpiled = pm.run(circ)
                if not isinstance(transpiled, list):
                    transpiled = [transpiled]
                sampler = SamplerV2(mode=backend)
                counts = sampler.run(transpiled, shots=shots).result()[0].join_data().get_counts()
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

    def run(self, circuit, observable=None, rem=None, qubits=None) -> float:
        """Run the circuit with ZNE and return the mitigated value.

        Measurements are stripped before passing to mitiq in both modes:

        - observable mode: mitiq adds its own measurements via
          ``observable.measure_in()``.
        - float mode: the executor re-adds measurements via ``measure_all()``
          if absent.

        This also lets mitiq correctly assess circuit length for the
        short-circuit warning.

        Args:
            circuit (QuantumCircuit): Circuit to execute. Measurements are
                handled internally by mitiq.
            observable (mitiq.Observable | None): Pauli observable to measure.
                ``None`` → P(|0…0⟩). Example:
                ``Observable(PauliString("ZZ", support=[0, 1]))``
            rem (ReadoutMitigation | None): Optional REM correction applied
                inside the executor.
            qubits (list[int] | None): Physical qubits; required when ``rem``
                is provided.

        Returns:
            float: Extrapolated ⟨observable⟩, or P(|0…0⟩) if
                ``observable`` is None.
        """
        zne = _require_mitiq_zne()

        executor = self._make_executor(rem=rem, qubits=qubits, observable=observable)

        kwargs = {}
        if self.factory is not None:
            kwargs["factory"] = self.factory
        else:
            # LinearFactory is more stable than Richardson with many scale factors
            # (Richardson fits a degree-(n-1) polynomial that can diverge with 4+ points).
            kwargs["factory"] = zne.inference.LinearFactory(self.scale_factors)
        if self.scale_noise is not None:
            kwargs["scale_noise"] = self.scale_noise
        if observable is not None:
            kwargs["observable"] = observable

        circuit = circuit.remove_final_measurements(inplace=False)

        result = zne.execute_with_zne(circuit, executor, **kwargs)
        return float(result.real) if hasattr(result, "real") else float(result)

    def run_unmitigated(self, circuit, observable=None, rem=None, qubits=None) -> float:
        """Run the circuit without mitigation (scale=1) for baseline comparison.

        Args:
            circuit (QuantumCircuit): Circuit to execute.
            observable (mitiq.Observable | None): Pauli observable to measure.
                ``None`` → P(|0…0⟩).
            rem (ReadoutMitigation | None): Optional REM correction applied
                inside the executor.
            qubits (list[int] | None): Physical qubits; required when ``rem``
                is provided.

        Returns:
            float: Raw ⟨observable⟩, or P(|0…0⟩) if ``observable`` is None.
        """
        executor = self._make_executor(rem=rem, qubits=qubits, observable=observable)
        if observable is not None:
            result = executor(circuit)
            return float(observable._expectation_from_measurements([result]).real)
        return executor(circuit)

    def run_scaled(self, circuit, observable=None) -> dict[float, float]:
        """Run the circuit at each scale factor and return the noise curve.

        Returns a mapping ``{scale_factor: expectation_value}``. Useful for
        inspecting the noise curve before extrapolation.

        Args:
            circuit (QuantumCircuit): Circuit to execute.
            observable (mitiq.Observable | None): Pauli observable to measure.
                ``None`` → P(|0…0⟩).

        Returns:
            dict[float, float]: Scale factor → expectation value at that noise
                level.
        """
        zne = _require_mitiq_zne()
        executor = self._make_executor(observable=observable)

        folded = [
            zne.scaling.fold_gates_at_random(circuit, s)
            if self.scale_noise is None
            else self.scale_noise(circuit, s)
            for s in self.scale_factors
        ]

        if observable is not None:
            values = [
                float(observable._expectation_from_measurements([executor(fc)]).real)
                for fc in folded
            ]
        else:
            values = [executor(fc) for fc in folded]

        return dict(zip(self.scale_factors, values))
