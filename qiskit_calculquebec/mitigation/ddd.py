"""
Digital Dynamical Decoupling (DDD) pour MonarQ
===============================================

Insère des séquences de portes dans les fenêtres inactives (idle windows)
du circuit pour découpler les qubits du bruit de déphasing.

Règles disponibles (mitiq.ddd.rules) :
  - ``'xx'``   : séquence X-X
  - ``'yy'``   : séquence Y-Y
  - ``'xyxy'`` : séquence X-Y-X-Y (recommandée en général)
"""


_VALID_RULES = ("xx", "yy", "xyxy")


def _require_mitiq_ddd():
    try:
        from mitiq.ddd import execute_with_ddd
        from mitiq.ddd.rules import xx, yy, xyxy
        return execute_with_ddd, {"xx": xx, "yy": yy, "xyxy": xyxy}
    except ImportError:
        raise ImportError(
            "mitiq est requis pour DDDMitigation.\n"
            "Installez-le avec : pip install mitiq\n"
            "ou : pip install qiskit-calculquebec[mitigation]"
        )


class DDDMitigation:
    """
    Digital Dynamical Decoupling (DDD) pour MonarQ.

    Parameters
    ----------
    backend : MonarQBackend
        Backend Calcul Québec.
    rule : str
        Règle DDD : ``'xx'``, ``'yy'`` ou ``'xyxy'`` (défaut).
    num_trials : int
        Nombre de répétitions pour moyenner le stochastique du placement,
        défaut 3.
    shots : int
        Shots par circuit, défaut 1024.

    Examples
    --------
    Observable par défaut — P(|0…0⟩) :

    >>> ddd = DDDMitigation(backend, rule='xyxy')
    >>> result = ddd.run(circuit)
    >>> print(f"Brut : {ddd.run_unmitigated(circuit):.4f}  DDD : {result:.4f}")

    Observable Pauli quelconque :

    >>> from mitiq import Observable, PauliString
    >>> obs = Observable(PauliString("ZZ", support=[0, 1]))  # ⟨Z₀Z₁⟩
    >>> result = ddd.run(circuit, observable=obs)

    Combinaison DDD + REM :

    >>> rem = ReadoutMitigation(backend, method='m3')
    >>> rem.cals_from_system()
    >>> # qubits doit être dérivé avec optimization_level=0 — le même niveau
    >>> # qu'utilise l'executor DDD en interne — pour que REM corrige
    >>> # les bons qubits physiques.
    >>> pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
    >>> t = pm.run(circuit)
    >>> physical_qubits = [t.layout.final_layout[q] for q in t.qubits] if t.layout and t.layout.final_layout else list(range(circuit.num_qubits))
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
            raise ValueError(f"rule doit être parmi {_VALID_RULES}, pas {rule!r}.")
        self.backend = backend
        self.rule = rule
        self.num_trials = num_trials
        self.shots = shots

    # ─────────────────────────────────────────────────────────────────────

    def _make_executor(self, rem=None, qubits=None, observable=None):
        """
        Retourne un executor compatible mitiq.ddd.

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
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import SamplerV2 as Sampler

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
                sampler = Sampler(mode=backend)
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
                # Mitiq peut passer un circuit sans mesures (après insertion DDD)
                circ = circuit.copy()
                if circ.num_clbits == 0:
                    circ.measure_all()

                pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
                transpiled = pm.run(circ)
                if not isinstance(transpiled, list):
                    transpiled = [transpiled]
                sampler = Sampler(mode=backend)
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
        Exécute le circuit avec DDD.

        Parameters
        ----------
        circuit : QuantumCircuit
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
            ⟨observable⟩ mitigé (moyenne sur ``num_trials``) ou P(|0…0⟩) si observable=None.
        """
        execute_with_ddd, rules = _require_mitiq_ddd()
        executor = self._make_executor(rem=rem, qubits=qubits, observable=observable)

        kwargs = {}
        if observable is not None:
            kwargs["observable"] = observable

        # Supprimer les mesures dans les deux modes :
        # - mode observable : mitiq ajoute ses propres mesures via observable.measure_in()
        # - mode float      : l'executor les réinsère via measure_all() si absent
        # Cela permet aussi à mitiq d'évaluer correctement la longueur du circuit
        # (ex. avertissement "circuit very short").
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
        """
        Exécute le circuit **sans** DDD pour comparaison.

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
