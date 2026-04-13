"""
Zero-Noise Extrapolation (ZNE) pour MonarQ
==========================================

Encapsule l'executor MonarQ et délègue l'extrapolation à mitiq.zne.
optimization_level=0 est **obligatoire** pour ZNE : le pass manager ne doit
pas modifier le circuit après le folding de bruit.
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
            "mitiq est requis pour ZNEMitigation.\n"
            "Installez-le avec : pip install mitiq\n"
            "ou : pip install qiskit-calculquebec[mitigation]"
        )


class ZNEMitigation:
    """
    Zero-Noise Extrapolation (ZNE) pour MonarQ.

    Parameters
    ----------
    backend : MonarQBackend
        Backend Calcul Québec.
    scale_factors : list[float]
        Facteurs de mise à l'échelle du bruit, défaut ``[1.0, 1.5, 2.0, 2.5, 3.0]``.
    factory : mitiq.zne.inference.Factory | None
        Méthode d'extrapolation. ``None`` → ``LinearFactory(scale_factors)`` (défaut).
        Richardson est plus précis en théorie mais diverge facilement avec 4+ points.
        Exemples : ``LinearFactory([1,2,3])``, ``RichardsonFactory([1,2,3])``,
        ``ExpFactory([1,2,3], asymptote=0.5)``.
    scale_noise : callable | None
        Méthode de folding. ``None`` → ``fold_gates_at_random`` (défaut mitiq).
        Alternatives : ``fold_global``.
    shots : int
        Nombre de shots par circuit, défaut 1024.

    Examples
    --------
    Observable par défaut — P(|0…0⟩) :

    >>> zne_mit = ZNEMitigation(backend, scale_factors=[1.0, 2.0, 3.0])
    >>> result = zne_mit.run(circuit)
    >>> print(f"Brut : {zne_mit.run_unmitigated(circuit):.4f}  Mitigé : {result:.4f}")

    Observable Pauli quelconque :

    >>> from mitiq import Observable, PauliString
    >>> obs = Observable(PauliString("ZZ", support=[0, 1]))  # ⟨Z₀Z₁⟩
    >>> result = zne_mit.run(circuit, observable=obs)

    Avec une factory personnalisée :

    >>> from mitiq.zne.inference import LinearFactory
    >>> from mitiq.zne.scaling import fold_global
    >>> zne_mit = ZNEMitigation(
    ...     backend,
    ...     factory=LinearFactory([1.0, 1.5, 2.0, 2.5, 3.0]),
    ...     scale_noise=fold_global,
    ... )
    >>> result = zne_mit.run(circuit)

    Combinaison ZNE + REM :

    >>> rem = ReadoutMitigation(backend, method='m3')
    >>> rem.cals_from_system()
    >>> # qubits doit être dérivé avec optimization_level=0 — le même niveau
    >>> # qu'utilise l'executor ZNE en interne — pour que REM corrige
    >>> # les bons qubits physiques.
    >>> pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
    >>> t = pm.run(circuit)
    >>> physical_qubits = [t.layout.final_layout[q] for q in t.qubits] if t.layout and t.layout.final_layout else list(range(circuit.num_qubits))
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
        """
        Retourne un executor compatible mitiq.zne pour ce backend.

        optimization_level=0 est obligatoire : le transpileur ne doit pas
        modifier le circuit après le folding appliqué par mitiq.

        Deux modes selon ``observable`` :

        - ``observable=None`` : retourne ``float`` — P(|0…0⟩).
        - ``observable`` fourni : retourne ``MeasurementResult`` (bitstrings bruts) ;
          mitiq calcule lui-même l'espérance via l'observable.

        Parameters
        ----------
        rem : ReadoutMitigation | None
            Si fourni, la correction REM est appliquée dans l'executor.
        qubits : list[int] | None
            Qubits physiques, requis si rem est fourni.
        observable : mitiq.Observable | None
            Si fourni, l'executor retourne MeasurementResult au lieu de float.
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
                counts = {"".join(k.split()): v for k, v in counts.items()}

                if rem is not None and qubits is None:
                    raise ValueError("qubits est requis quand rem est fourni.")

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
                # Mitiq supprime les mesures avant folding — on les réinsère si nécessaire
                circ = circuit.copy()
                if circ.num_clbits == 0:
                    circ.measure_all()

                pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
                transpiled = pm.run(circ)
                if not isinstance(transpiled, list):
                    transpiled = [transpiled]
                sampler = SamplerV2(mode=backend)
                counts = sampler.run(transpiled, shots=shots).result()[0].join_data().get_counts()
                counts = {"".join(k.split()): v for k, v in counts.items()}
                n = circuit.num_qubits

                if rem is not None and qubits is None:
                    raise ValueError("qubits est requis quand rem est fourni.")

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
        """
        Exécute le circuit avec ZNE et retourne la valeur mitigée.

        Parameters
        ----------
        circuit : QuantumCircuit
            Circuit à exécuter (sans mesures — mitiq les gère en interne).
        observable : mitiq.Observable | None
            Observable Pauli à mesurer. ``None`` → P(|0…0⟩).
            Exemple : ``Observable(PauliString("ZZ", support=[0, 1]))``
        rem : ReadoutMitigation | None
            Optionnel : applique aussi une correction REM dans l'executor.
        qubits : list[int] | None
            Qubits physiques, requis si ``rem`` est fourni.

        Returns
        -------
        float
            Valeur extrapolée ⟨observable⟩ ou P(|0…0⟩) si observable=None.
        """
        zne = _require_mitiq_zne()

        executor = self._make_executor(rem=rem, qubits=qubits, observable=observable)

        kwargs = {}
        if self.factory is not None:
            kwargs["factory"] = self.factory
        else:
            # LinearFactory est plus stable que Richardson pour des scale_factors
            # nombreux (Richardson degré n-1 peut diverger avec 4+ points).
            kwargs["factory"] = zne.inference.LinearFactory(self.scale_factors)
        if self.scale_noise is not None:
            kwargs["scale_noise"] = self.scale_noise
        if observable is not None:
            kwargs["observable"] = observable

        # Supprimer les mesures dans les deux modes :
        # - mode observable : mitiq ajoute ses propres mesures via observable.measure_in()
        # - mode float      : l'executor les réinsère via measure_all() si absent
        # Cela permet aussi à mitiq d'évaluer correctement la longueur du circuit
        # (ex. avertissement "circuit very short").
        circuit = circuit.remove_final_measurements(inplace=False)

        result = zne.execute_with_zne(circuit, executor, **kwargs)
        return float(result.real) if hasattr(result, "real") else float(result)

    def run_unmitigated(self, circuit, observable=None, rem=None, qubits=None) -> float:
        """
        Exécute le circuit **sans** mitigation (scale=1) pour comparaison.

        Parameters
        ----------
        observable : mitiq.Observable | None
            Observable Pauli à mesurer. ``None`` → P(|0…0⟩).
        rem : ReadoutMitigation | None
            Optionnel : applique aussi une correction REM.
        qubits : list[int] | None
            Qubits physiques, requis si ``rem`` est fourni.

        Returns
        -------
        float
            ⟨observable⟩ brut ou P(|0…0⟩) si observable=None.
        """
        executor = self._make_executor(rem=rem, qubits=qubits, observable=observable)
        if observable is not None:
            result = executor(circuit)
            return float(observable._expectation_from_measurements([result]).real)
        return executor(circuit)

    def run_scaled(self, circuit, observable=None) -> dict[float, float]:
        """
        Exécute le circuit à chaque facteur d'échelle et retourne le mapping
        ``{scale_factor: expectation_value}``.

        Utile pour inspecter la courbe de bruit avant extrapolation.

        Parameters
        ----------
        observable : mitiq.Observable | None
            Observable Pauli à mesurer. ``None`` → P(|0…0⟩).
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
