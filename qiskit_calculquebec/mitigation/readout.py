"""
Readout Error Mitigation (REM) pour MonarQ
==========================================

Deux méthodes disponibles :

``method='matrix'``
    Matrice de confusion inverse (approche mitiq).
    Construit la matrice 2ⁿ × 2ⁿ complète en produit tensoriel des matrices
    de calibration locales, puis applique l'inverse.
    Simple et exacte, mais limitée à ~12 qubits (mémoire exponentielle).

``method='m3'``
    Matrix-free Measurement Mitigation (mthree).
    Travaille sur la sous-matrice réduite aux seuls bitstrings observés —
    scalable jusqu'à 24 qubits et au-delà.
    Recommandée dès que le nombre de qubits mesurés dépasse ~10.

Dans les deux cas, les fidélités de calibration P(0|0) et P(1|1) sont lues
**directement depuis le benchmark Anyon** via l'API Calcul Québec, sans
soumettre de circuits de calibration supplémentaires sur le hardware.
"""

from __future__ import annotations

import warnings
import datetime
import logging
from functools import reduce

import numpy as np

from qiskit_calculquebec.API.adapter import ApiAdapter

logger = logging.getLogger(__name__)

_DEFAULT_FIDELITY = 1.0 - 2e-2  # valeur par défaut si absent du benchmark


# ── imports optionnels ─────────────────────────────────────────────────────

def _require_mitiq():
    try:
        import mitiq  # noqa: F401
        from mitiq import MeasurementResult
        from mitiq.rem.inverse_confusion_matrix import mitigate_measurements
        return MeasurementResult, mitigate_measurements
    except ImportError:
        raise ImportError(
            "mitiq est requis pour method='matrix'.\n"
            "Installez-le avec : pip install mitiq\n"
            "ou : pip install qiskit-calculquebec[mitigation]"
        )


def _require_mthree():
    try:
        from mthree.direct import direct_solver as _direct_solve
        from mthree.direct import reduced_cal_matrix as _cal_matrix
        from mthree.iterative import iterative_solver as _iterative_solver
        from mthree.classes import QuasiCollection
        from mthree.exceptions import M3Error
        import psutil
        return _direct_solve, _cal_matrix, _iterative_solver, QuasiCollection, M3Error, psutil
    except ImportError:
        raise ImportError(
            "mthree et psutil sont requis pour method='m3'.\n"
            "Installez-les avec : pip install mthree psutil\n"
            "ou : pip install qiskit-calculquebec[mitigation]"
        )


# ══════════════════════════════════════════════════════════════════════════
class ReadoutMitigation:
    """
    Mitigation des erreurs de lecture (REM) pour MonarQ.

    Parameters
    ----------
    backend : MonarQBackend
        Backend Calcul Québec initialisé avec un ``ApiClient``.
    method : str
        ``'matrix'`` (mitiq, exact, limité ~12 qubits) ou
        ``'m3'`` (mthree, scalable jusqu'à 24 qubits). Défaut : ``'m3'``.
    iter_threshold : int
        (M3 uniquement) Seuil de bitstrings distincts au-delà duquel le
        solveur itératif (GMRES) est préféré au solveur direct. Défaut : 4096.

    Examples
    --------
    **Méthode M3 (recommandée pour MonarQ 24 qubits) :**

    >>> mit = ReadoutMitigation(backend, method='m3')
    >>> mit.cals_from_system()
    >>> quasi = mit.apply_correction(counts_raw, qubits=physical_qubits)
    >>> counts_mit = {k: int(round(v * shots))
    ...               for k, v in quasi.nearest_probability_distribution().items()
    ...               if round(v * shots) > 0}

    **Méthode matrice (compatible mitiq, petits circuits) :**

    >>> mit = ReadoutMitigation(backend, method='matrix')
    >>> mit.cals_from_system()
    >>> counts_mit = mit.apply_correction(counts_raw, qubits=physical_qubits)
    """

    def __init__(self, backend, method: str = "m3", iter_threshold: int = 4096):
        if method not in ("matrix", "m3"):
            raise ValueError(f"method doit être 'matrix' ou 'm3', pas {method!r}.")
        self.backend = backend
        self.method = method
        self.iter_threshold = iter_threshold

        # Calibrations par qubit physique (None = non calibré)
        self.single_qubit_cals: list | None = None
        self.num_qubits: int = backend.target.num_qubits
        self.faulty_qubits: list = []
        self.cal_timestamp: str | None = None

    # ─────────────────────────────────────────────────────────────────────
    # Calibration — commune aux deux méthodes
    # ─────────────────────────────────────────────────────────────────────

    def cals_from_system(self, qubits: list[int] | None = None):
        """
        Charge P(0|0) et P(1|1) depuis le benchmark Anyon (API Calcul Québec).

        Aucun circuit de calibration n'est soumis au hardware.

        Parameters
        ----------
        qubits : list[int] | None
            Qubits physiques à calibrer. ``None`` → tous les qubits du backend.
        """
        if qubits is None:
            qubits = list(range(self.num_qubits))

        machine_name = self.backend._client.machine_name
        benchmark = ApiAdapter.get_benchmark(machine_name)
        qubits_data = benchmark["resultsPerDevice"]["qubits"]

        self.single_qubit_cals = [None] * self.num_qubits

        for q in qubits:
            qb = qubits_data.get(str(q), {})
            p0 = qb.get("parallelReadoutState0Fidelity", _DEFAULT_FIDELITY)
            p1 = qb.get("parallelReadoutState1Fidelity", _DEFAULT_FIDELITY)
            # Matrice de calibration 2×2 :
            #   col 0 → préparé |0⟩ : [P(0|0), P(1|0)]
            #   col 1 → préparé |1⟩ : [P(0|1), P(1|1)]
            self.single_qubit_cals[q] = np.array(
                [[p0,       1.0 - p1],
                 [1.0 - p0, p1      ]],
                dtype=np.float64,
            )

        self.cal_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.faulty_qubits = _faulty_qubit_checker(self.single_qubit_cals)

        if self.faulty_qubits:
            warnings.warn(
                f"Qubits avec calibration dégradée (P(0|1) ≥ P(0|0)) : "
                f"{self.faulty_qubits}"
            )

        logger.info(
            "Calibration chargée pour %d qubits depuis le benchmark Anyon (%s).",
            len(qubits), machine_name,
        )

    def cals_from_matrices(self, matrices: list):
        """
        Initialise les calibrations depuis une liste de matrices NumPy 2×2.

        Parameters
        ----------
        matrices : list[np.ndarray | None]
            Liste de longueur ``num_qubits``. ``None`` pour les qubits non calibrés.
        """
        matrices = list(matrices)
        if len(matrices) != self.num_qubits:
            raise ValueError(
                f"Longueur de la liste ({len(matrices)}) ≠ num_qubits ({self.num_qubits})."
            )
        self.single_qubit_cals = [
            np.asarray(m, dtype=np.float64) if m is not None else None
            for m in matrices
        ]
        self.faulty_qubits = _faulty_qubit_checker(self.single_qubit_cals)

    def readout_fidelity(self, qubits: list[int] | None = None) -> list:
        """
        Retourne la fidélité de lecture (P(0|0), P(1|1), moyenne) par qubit.

        Returns
        -------
        list[dict | None]
            Chaque élément est ``{'p00': float, 'p11': float, 'mean': float}``
            ou ``None`` si le qubit n'est pas calibré.
        """
        if self.single_qubit_cals is None:
            raise RuntimeError("Mitigateur non calibré. Appelez cals_from_system() d'abord.")
        if qubits is None:
            qubits = range(self.num_qubits)
        result = []
        for q in qubits:
            cal = self.single_qubit_cals[q]
            if cal is None:
                result.append(None)
            else:
                p00, p11 = float(cal[0, 0]), float(cal[1, 1])
                result.append({"p00": p00, "p11": p11, "mean": (p00 + p11) / 2})
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Correction — dispatch selon method
    # ─────────────────────────────────────────────────────────────────────

    def apply_correction(self, counts, qubits: list[int], **kwargs):
        """
        Applique la correction REM aux counts bruts.

        Parameters
        ----------
        counts : dict
            Counts Qiskit (bitstrings little-endian → nombre de shots).
        qubits : list[int]
            Qubits physiques correspondant aux bits des bitstrings
            (dans le même ordre que les bits du bitstring).
        **kwargs
            Paramètres additionnels transmis à la méthode sous-jacente.

            *Pour method='m3' :*
              - ``distance`` (int | None) : distance de Hamming max, défaut ``min(n, 3)``.
              - ``solver`` (str) : ``'auto'``, ``'direct'`` ou ``'iterative'``.
              - ``max_iter`` (int) : itérations GMRES max, défaut 25.
              - ``tol`` (float) : tolérance GMRES, défaut 1e-4.
              - ``details`` (bool) : retourne aussi les métriques du solveur.

        Returns
        -------
        *Pour method='matrix'* : ``dict`` de counts corrigés (même format que l'entrée).
        *Pour method='m3'* : ``QuasiDistribution`` (mthree).
            Convertissez avec ``quasi.nearest_probability_distribution()``.

        Raises
        ------
        RuntimeError
            Si le mitigateur n'est pas calibré.
        """
        if self.single_qubit_cals is None:
            raise RuntimeError("Mitigateur non calibré. Appelez cals_from_system() d'abord.")

        missing = [q for q in qubits if self.single_qubit_cals[q] is None]
        if missing:
            raise RuntimeError(
                f"Qubits non calibrés : {missing}. "
                "Appelez cals_from_system(qubits=...) avec ces qubits."
            )

        bad = set(qubits) & set(self.faulty_qubits)
        if bad:
            warnings.warn(f"Qubits défectueux utilisés dans la correction : {sorted(bad)}")

        if self.method == "matrix":
            return self._apply_matrix(counts, qubits)
        else:
            return self._apply_m3(counts, qubits, **kwargs)

    # ─────────────────────────────────────────────────────────────────────
    # Méthode matrice (mitiq)
    # ─────────────────────────────────────────────────────────────────────

    def _apply_matrix(self, counts: dict, qubits: list[int]) -> dict:
        """
        REM par matrice de confusion inverse (mitiq).

        Construit A = ⊗ A_i puis applique A⁺ aux counts.
        Coût mémoire : O(4ⁿ) → pratique jusqu'à ~12 qubits.
        """
        MeasurementResult, mitigate_measurements = _require_mitiq()

        n = len(qubits)
        if n > 15:
            warnings.warn(
                f"method='matrix' avec {n} qubits : la matrice {2**n}×{2**n} "
                "risque de saturer la mémoire. Considérez method='m3'."
            )

        # Matrice inverse de confusion globale (produit tensoriel des pseudo-inverses)
        inv_confusion = self._build_inv_confusion_matrix(qubits)

        # Conversion counts → MeasurementResult mitiq (big-endian)
        bitstrings = []
        for bitstring, count in counts.items():
            bits = [int(b) for b in reversed(bitstring)]
            bitstrings.extend([bits] * count)
        noisy_result = MeasurementResult(np.array(bitstrings, dtype=int))

        # Application de la correction
        mitigated_result = mitigate_measurements(noisy_result, inv_confusion)

        # Reconversion en counts Qiskit (little-endian)
        corrected_counts: dict = {}
        for bits in mitigated_result.result:
            bs = "".join(str(b) for b in reversed(bits))
            corrected_counts[bs] = corrected_counts.get(bs, 0) + 1

        return corrected_counts

    def _build_inv_confusion_matrix(self, qubits: list[int]) -> np.ndarray:
        """
        Produit tensoriel des pseudo-inverses des matrices de calibration locales.

        A⁺ = pinv(A_q0) ⊗ pinv(A_q1) ⊗ … ⊗ pinv(A_qn-1)
        """
        pinv_matrices = [np.linalg.pinv(self.single_qubit_cals[q]) for q in qubits]
        return reduce(np.kron, pinv_matrices)

    def get_confusion_matrix(self, qubits: list[int]) -> np.ndarray:
        """
        Retourne la matrice de confusion directe (non inversée) pour les qubits donnés.

        Utile pour visualisation (heatmap).
        """
        if self.single_qubit_cals is None:
            raise RuntimeError("Mitigateur non calibré.")
        mats = [self.single_qubit_cals[q] for q in qubits]
        return reduce(np.kron, mats)

    def get_inv_confusion_matrix(self, qubits: list[int]) -> np.ndarray:
        """
        Retourne la matrice inverse de confusion (produit tensoriel des pseudo-inverses).

        Utile pour visualisation ou réutilisation.
        """
        if self.single_qubit_cals is None:
            raise RuntimeError("Mitigateur non calibré.")
        return self._build_inv_confusion_matrix(qubits)

    # ─────────────────────────────────────────────────────────────────────
    # Méthode M3 (mthree)
    # ─────────────────────────────────────────────────────────────────────

    def _apply_m3(
        self,
        counts: dict,
        qubits: list[int],
        distance: int | None = None,
        solver: str = "auto",
        max_iter: int = 25,
        tol: float = 1e-4,
        details: bool = False,
        return_mitigation_overhead: bool = False,
    ):
        """
        REM via M3 (Matrix-free Measurement Mitigation, mthree).

        Travaille sur la sous-matrice réduite aux bitstrings observés.
        Scalable jusqu'à 24 qubits et au-delà.
        """
        from time import perf_counter

        _direct_solve, _cal_matrix, _iterative_solver, QuasiCollection, M3Error, psutil = (
            _require_mthree()
        )

        counts = dict(counts)
        shots = sum(counts.values())
        num_bits = len(qubits)
        num_elems = len(counts)

        if distance is None:
            distance = min(num_bits, 3)
        elif distance == -1:
            distance = num_bits

        bitstring_len = len(next(iter(counts)))
        if bitstring_len != num_bits:
            raise ValueError(
                f"Longueur bitstring ({bitstring_len}) ≠ len(qubits) ({num_bits})."
            )

        # Choix automatique du solveur
        if solver == "auto":
            free_gb = psutil.virtual_memory().available / 1024**3
            if num_elems <= self.iter_threshold and (
                (num_elems**2 + num_elems) * 8 / 1024**3 < free_gb / 2
            ):
                solver = "direct"
            else:
                solver = "iterative"
            logger.debug("M3 solveur auto-sélectionné : %s", solver)

        if solver == "direct":
            st = perf_counter()
            mit_counts, col_norms, gamma = _direct_solve(
                self, counts, qubits, distance, return_mitigation_overhead
            )
            dur = perf_counter() - st
            mit_counts.shots = shots
            if gamma is not None:
                mit_counts.mitigation_overhead = gamma * gamma
            if details:
                return mit_counts, {
                    "method": "direct", "time": dur,
                    "dimension": num_elems, "col_norms": col_norms,
                }
            return mit_counts

        elif solver == "iterative":
            iter_count = np.zeros(1, dtype=int)

            def _cb(_):
                iter_count[0] += 1

            if details:
                st = perf_counter()
                mit_counts, col_norms, gamma = _iterative_solver(
                    self, counts, qubits, distance, tol, max_iter, 1, _cb,
                    return_mitigation_overhead,
                )
                dur = perf_counter() - st
                mit_counts.shots = shots
                if gamma is not None:
                    mit_counts.mitigation_overhead = gamma * gamma
                return mit_counts, {
                    "method": "iterative", "time": dur,
                    "dimension": num_elems, "iterations": iter_count[0],
                    "col_norms": col_norms,
                }
            mit_counts, gamma = _iterative_solver(
                self, counts, qubits, distance, tol, max_iter, 0, _cb,
                return_mitigation_overhead,
            )
            mit_counts.shots = shots
            if gamma is not None:
                mit_counts.mitigation_overhead = gamma * gamma
            return mit_counts

        else:
            raise ValueError(f"solver invalide : {solver!r}. Choisir 'auto', 'direct' ou 'iterative'.")

    # ─────────────────────────────────────────────────────────────────────
    # Interface interne pour les solveurs M3 (appelée par mthree)
    # ─────────────────────────────────────────────────────────────────────

    def _form_cals(self, qubits) -> np.ndarray:
        """
        Tableau 1-D de calibrations au format attendu par les solveurs mthree.

        Format : [P00_q(n-1), P10_q(n-1), P01_q(n-1), P11_q(n-1), …]
        (qubits en ordre reversed pour correspondre à l'indexation interne M3).
        """
        qubits = np.asarray(qubits, dtype=int)
        cals = np.zeros(4 * len(qubits), dtype=np.float32)
        for kk, qubit in enumerate(qubits[::-1]):
            cals[4 * kk: 4 * kk + 4] = self.single_qubit_cals[qubit].astype(np.float32).ravel()
        return cals

    def reduced_cal_matrix(self, counts, qubits, distance=None):
        """
        (M3) Retourne la sous-matrice de calibration réduite aux bitstrings observés.

        Parameters
        ----------
        counts : dict
        qubits : list[int]
        distance : int | None

        Returns
        -------
        (ndarray, dict)
        """
        _direct_solve, _cal_matrix, *_ = _require_mthree()
        return _cal_matrix(self, counts, qubits, distance)


# ─────────────────────────────────────────────────────────────────────────
# Utilitaire interne
# ─────────────────────────────────────────────────────────────────────────

def _faulty_qubit_checker(cals: list) -> list:
    """Identifie les qubits dont P(0|1) ≥ P(0|0) (calibration inversée)."""
    return [
        idx for idx, cal in enumerate(cals)
        if cal is not None and cal[0, 1] >= cal[0, 0]
    ]
