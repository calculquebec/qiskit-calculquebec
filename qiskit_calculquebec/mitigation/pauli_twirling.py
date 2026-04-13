"""
Pauli Twirling (PT) pour MonarQ
================================

Le Pauli Twirling insère des paires de portes de Pauli aléatoires autour
des portes 2-qubits (CNOT, CZ) pour convertir un canal de bruit arbitraire
en un canal de dépolarisation, plus simple à traiter par d'autres techniques.

Peut être combiné avec ZNE (PT + ZNE) pour une mitigation plus poussée.
"""

import numpy as np


def _require_mitiq_pt():
    try:
        from mitiq.pt import generate_pauli_twirl_variants
        return generate_pauli_twirl_variants
    except ImportError:
        raise ImportError(
            "mitiq est requis pour PauliTwirlingMitigation.\n"
            "Installez-le avec : pip install mitiq\n"
            "ou : pip install qiskit-calculquebec[mitigation]"
        )


def _require_mitiq_zne():
    try:
        from mitiq import zne
        return zne
    except ImportError:
        raise ImportError(
            "mitiq est requis pour PT + ZNE.\n"
            "Installez-le avec : pip install mitiq\n"
            "ou : pip install qiskit-calculquebec[mitigation]"
        )


class PauliTwirlingMitigation:
    """
    Pauli Twirling (PT) pour MonarQ, avec option ZNE.

    Parameters
    ----------
    backend : MonarQBackend
        Backend Calcul Québec.
    num_variants : int
        Nombre de variantes twirled à moyenner. Plus c'est élevé, plus la
        variance est réduite. Défaut : 10.
        Coût : ``num_variants × shots`` exécutions.
    shots : int
        Shots par variante, défaut 1024.

    Examples
    --------
    **PT seul :**

    >>> pt = PauliTwirlingMitigation(backend, num_variants=10)
    >>> result = pt.run(circuit)

    **PT + ZNE :**

    >>> pt = PauliTwirlingMitigation(backend, num_variants=10)
    >>> result = pt.run_with_zne(circuit, scale_factors=[1.0, 2.0, 3.0])

    **Toutes configurations :**

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
        """
        Executor simple : transpile + exécute + retourne P(|0…0⟩).

        Parameters
        ----------
        rem : ReadoutMitigation | None
            Si fourni, la correction REM est appliquée.
        qubits : list[int] | None
            Qubits physiques, requis si rem est fourni.
        """
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import SamplerV2 as Sampler

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

    def _make_pt_executor(self, rem=None, qubits=None):
        """
        Executor PT : génère ``num_variants`` variantes twirled,
        les exécute et retourne la moyenne.
        Utilisé comme executor dans ZNE.

        Parameters
        ----------
        rem : ReadoutMitigation | None
            Si fourni, la correction REM est appliquée dans chaque variante.
        qubits : list[int] | None
            Qubits physiques, requis si rem est fourni.
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
        """
        Exécute le circuit avec Pauli Twirling seul.

        Parameters
        ----------
        circuit : QuantumCircuit
        rem : ReadoutMitigation | None
            Optionnel : applique aussi une correction REM.
        qubits : list[int] | None
            Qubits physiques, requis si ``rem`` est fourni.

        Returns
        -------
        float
            P(|0…0⟩) moyenné sur ``num_variants`` variantes.
        """
        return self._make_pt_executor(rem=rem, qubits=qubits)(circuit)

    def run_unmitigated(self, circuit, rem=None, qubits=None) -> float:
        """
        Exécute le circuit sans twirling pour comparaison.

        Parameters
        ----------
        rem : ReadoutMitigation | None
            Optionnel : applique aussi une correction REM.
        qubits : list[int] | None
            Qubits physiques, requis si ``rem`` est fourni.
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
        """
        PT + ZNE : combine twirling et extrapolation à bruit nul.

        Parameters
        ----------
        circuit : QuantumCircuit
        scale_factors : list[float] | None
            Ignoré si ``factory`` est fourni. Défaut : [1.0, 1.5, 2.0, 2.5, 3.0].
        factory : mitiq.zne.inference.Factory | None
            ``None`` → ``LinearFactory(scale_factors)``.
            LinearFactory est plus stable que Richardson avec 4+ points.
        scale_noise : callable | None
            ``None`` → ``fold_gates_at_random``.
        rem : ReadoutMitigation | None
            Optionnel : applique aussi une correction REM.
        qubits : list[int] | None
            Qubits physiques, requis si ``rem`` est fourni.

        Returns
        -------
        float
            Valeur PT + ZNE extrapolée.
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

        # Supprimer les mesures — mitiq les gère en interne via le folding
        circuit = circuit.remove_final_measurements(inplace=False)

        result = zne.execute_with_zne(circuit, pt_executor, **kwargs)
        return float(result.real) if hasattr(result, "real") else float(result)

    def run_variants(self, circuit, rem=None, qubits=None) -> list[float]:
        """
        Retourne les ``num_variants`` valeurs individuelles (sans moyenner).

        Utile pour inspecter la variance du twirling.

        Parameters
        ----------
        rem : ReadoutMitigation | None
            Optionnel : applique aussi une correction REM.
        qubits : list[int] | None
            Qubits physiques, requis si ``rem`` est fourni.
        """
        generate_pauli_twirl_variants = _require_mitiq_pt()
        base_executor = self._make_base_executor(rem=rem, qubits=qubits)
        variants = generate_pauli_twirl_variants(circuit, num_circuits=self.num_variants)
        return [base_executor(v) for v in variants]
