"""
Digital Dynamical Decoupling (DDD) for MonarQ
==============================================

Inserts pulse sequences into idle windows of the circuit to decouple
qubits from dephasing noise.

Available rules (mitiq.ddd.rules):
  - ``'xx'``   : X-X sequence
  - ``'yy'``   : Y-Y sequence
  - ``'xyxy'`` : X-Y-X-Y sequence (recommended in general)
"""


try:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2
except ImportError:
    generate_preset_pass_manager = None  # type: ignore[assignment]
    SamplerV2 = None  # type: ignore[assignment,misc]


_VALID_RULES = ("xx", "yy", "xyxy")


def _require_mitiq_ddd():
    try:
        from mitiq.ddd import execute_with_ddd
        from mitiq.ddd.rules import xx, yy, xyxy
        return execute_with_ddd, {"xx": xx, "yy": yy, "xyxy": xyxy}
    except ImportError:
        raise ImportError(
            "mitiq is required for DDDMitigation.\n"
            "Install it with: pip install mitiq\n"
            "or: pip install qiskit-calculquebec[mitigation]"
        )


class DDDMitigation:
    """Digital Dynamical Decoupling (DDD) for MonarQ.

    Inserts decoupling gate sequences into idle windows to suppress
    dephasing noise. Results are averaged over ``num_trials`` randomized
    placements.

    Args:
        backend (MonarQBackend): Calcul Québec backend.
        rule (str): DDD rule: ``'xx'``, ``'yy'``, or ``'xyxy'`` (default).
        num_trials (int): Number of repetitions to average over the stochastic
            placement. Default: 3.
        shots (int): Shots per circuit. Default: 1024.

    Examples:
        Default observable — P(|0…0⟩):

        >>> ddd = DDDMitigation(backend, rule='xyxy')
        >>> result = ddd.run(circuit)
        >>> print(f"Raw: {ddd.run_unmitigated(circuit):.4f}  DDD: {result:.4f}")

        Arbitrary Pauli observable:

        >>> from mitiq import Observable, PauliString
        >>> obs = Observable(PauliString("ZZ", support=[0, 1]))  # <Z0 Z1>
        >>> result = ddd.run(circuit, observable=obs)

        DDD combined with REM:

        >>> rem = ReadoutMitigation(backend, method='m3')
        >>> rem.cals_from_system()
        >>> pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
        >>> t = pm.run(circuit)
        >>> physical_qubits = (
        ...     [t.layout.final_layout[q] for q in t.qubits]
        ...     if t.layout and t.layout.final_layout
        ...     else list(range(circuit.num_qubits))
        ... )
        >>> ddd = DDDMitigation(backend, rule='xyxy')
        >>> result = ddd.run(circuit, rem=rem, qubits=physical_qubits)
    """

    def __init__(
        self,
        backend,
        rule: str = "xyxy",
        num_trials: int = 3,
        shots: int = 1024,
    ):
        if rule not in _VALID_RULES:
            raise ValueError(f"rule must be one of {_VALID_RULES}, got {rule!r}.")
        self.backend = backend
        self.rule = rule
        self.num_trials = num_trials
        self.shots = shots

    # ─────────────────────────────────────────────────────────────────────

    def _make_executor(self, rem=None, qubits=None, observable=None):
        """Build a mitiq-compatible executor for this backend.

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
                # mitiq may pass a circuit without measurements after DDD insertion — re-add them
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
        """Run the circuit with DDD and return the mitigated value.

        Measurements are stripped before passing to mitiq in both modes:

        - observable mode: mitiq adds its own measurements via
          ``observable.measure_in()``.
        - float mode: the executor re-adds measurements via ``measure_all()``
          if absent.

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
            float: Mitigated ⟨observable⟩ (averaged over ``num_trials``), or
                P(|0…0⟩) if ``observable`` is None.
        """
        execute_with_ddd, rules = _require_mitiq_ddd()
        executor = self._make_executor(rem=rem, qubits=qubits, observable=observable)

        kwargs = {}
        if observable is not None:
            kwargs["observable"] = observable

        circuit = circuit.remove_final_measurements(inplace=False)

        result = execute_with_ddd(
            circuit=circuit,
            executor=executor,
            rule=rules[self.rule],
            num_trials=self.num_trials,
            **kwargs,
        )
        return float(result.real) if hasattr(result, "real") else float(result)

    def run_unmitigated(self, circuit, observable=None, rem=None, qubits=None) -> float:
        """Run the circuit without DDD for baseline comparison.

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
