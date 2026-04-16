"""
Readout Error Mitigation (REM) for MonarQ
==========================================

Two methods are available:

``method='matrix'``
    Inverse confusion matrix (mitiq approach).
    Builds the full 2ⁿ × 2ⁿ matrix as the tensor product of local
    calibration matrices, then applies its pseudo-inverse.
    Simple and exact, but limited to ~12 qubits due to exponential memory.

``method='m3'``
    Matrix-free Measurement Mitigation (mthree).
    Works on the reduced sub-matrix of observed bitstrings only —
    scalable to 24 qubits and beyond.
    Recommended when the number of measured qubits exceeds ~10.

In both cases, the calibration fidelities P(0|0) and P(1|1) are read
directly from the Anyon benchmark via the Calcul Québec API, without
submitting additional calibration circuits to the hardware.
"""

from __future__ import annotations

import warnings
import datetime
import logging
from functools import reduce

import numpy as np

from qiskit_calculquebec.API.adapter import ApiAdapter

logger = logging.getLogger(__name__)

_DEFAULT_FIDELITY = 1.0 - 2e-2  # default fidelity used when benchmark data is missing


# ── optional imports ───────────────────────────────────────────────────────

def _require_mitiq():
    try:
        import mitiq  # noqa: F401
        from mitiq import MeasurementResult
        from mitiq.rem.inverse_confusion_matrix import mitigate_measurements
        return MeasurementResult, mitigate_measurements
    except ImportError:
        raise ImportError(
            "mitiq is required for method='matrix'.\n"
            "Install it with: pip install mitiq\n"
            "or: pip install qiskit-calculquebec[mitigation]"
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
            "mthree and psutil are required for method='m3'.\n"
            "Install them with: pip install mthree psutil\n"
            "or: pip install qiskit-calculquebec[mitigation]"
        )


# ══════════════════════════════════════════════════════════════════════════
class ReadoutMitigation:
    """Readout Error Mitigation (REM) for MonarQ.

    Calibration data (P(0|0) and P(1|1) per qubit) is loaded from the
    Anyon benchmark via the Calcul Québec API — no calibration circuits
    are submitted to the hardware.

    Args:
        backend (MonarQBackend): Calcul Québec backend initialized with an
            ``ApiClient``.
        method (str): ``'matrix'`` (mitiq, exact, limited to ~12 qubits) or
            ``'m3'`` (mthree, scalable to 24+ qubits). Default: ``'m3'``.
        iter_threshold (int): (M3 only) Number of distinct bitstrings above
            which the iterative solver (GMRES) is preferred over the direct
            solver. Default: 4096.

    Examples:
        M3 method (recommended for MonarQ 24 qubits):

        >>> mit = ReadoutMitigation(backend, method='m3')
        >>> mit.cals_from_system()
        >>> quasi = mit.apply_correction(counts_raw, qubits=physical_qubits)
        >>> counts_mit = {k: int(round(v * shots))
        ...               for k, v in quasi.nearest_probability_distribution().items()
        ...               if round(v * shots) > 0}

        Matrix method (mitiq-compatible, small circuits):

        >>> mit = ReadoutMitigation(backend, method='matrix')
        >>> mit.cals_from_system()
        >>> counts_mit = mit.apply_correction(counts_raw, qubits=physical_qubits)
    """

    def __init__(self, backend, method: str = "m3", iter_threshold: int = 4096):
        if method not in ("matrix", "m3"):
            raise ValueError(f"method must be 'matrix' or 'm3', got {method!r}.")
        self.backend = backend
        self.method = method
        self.iter_threshold = iter_threshold

        # Per-physical-qubit calibration matrices (None = not calibrated)
        self.single_qubit_cals: list | None = None
        self.num_qubits: int = backend.target.num_qubits
        self.faulty_qubits: list = []
        self.cal_timestamp: str | None = None

    # ─────────────────────────────────────────────────────────────────────
    # Calibration — shared by both methods
    # ─────────────────────────────────────────────────────────────────────

    def cals_from_system(self, qubits: list[int] | None = None):
        """Load P(0|0) and P(1|1) from the Anyon benchmark (Calcul Québec API).

        No calibration circuits are submitted to the hardware.

        Args:
            qubits (list[int] | None): Physical qubits to calibrate. ``None``
                → all qubits on the backend.
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
            # 2×2 calibration matrix:
            #   col 0 → prepared |0⟩: [P(0|0), P(1|0)]
            #   col 1 → prepared |1⟩: [P(0|1), P(1|1)]
            self.single_qubit_cals[q] = np.array(
                [[p0,       1.0 - p1],
                 [1.0 - p0, p1      ]],
                dtype=np.float64,
            )

        self.cal_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.faulty_qubits = _faulty_qubit_checker(self.single_qubit_cals)

        if self.faulty_qubits:
            warnings.warn(
                f"Qubits with degraded calibration (P(0|1) >= P(0|0)): "
                f"{self.faulty_qubits}"
            )

        logger.info(
            "Calibration loaded for %d qubits from the Anyon benchmark (%s).",
            len(qubits), machine_name,
        )

    def cals_from_matrices(self, matrices: list):
        """Initialize calibration from a list of 2×2 NumPy matrices.

        Args:
            matrices (list[np.ndarray | None]): List of length ``num_qubits``.
                Use ``None`` for uncalibrated qubits.
        """
        matrices = list(matrices)
        if len(matrices) != self.num_qubits:
            raise ValueError(
                f"List length ({len(matrices)}) != num_qubits ({self.num_qubits})."
            )
        self.single_qubit_cals = [
            np.asarray(m, dtype=np.float64) if m is not None else None
            for m in matrices
        ]
        self.faulty_qubits = _faulty_qubit_checker(self.single_qubit_cals)

    def readout_fidelity(self, qubits: list[int] | None = None) -> list:
        """Return the readout fidelity (P(0|0), P(1|1), mean) per qubit.

        Args:
            qubits (list[int] | None): Qubits to query. ``None`` → all qubits.

        Returns:
            list[dict | None]: Each entry is
                ``{'p00': float, 'p11': float, 'mean': float}``, or ``None``
                if the qubit is not calibrated.

        Raises:
            RuntimeError: If calibration has not been loaded yet.
        """
        if self.single_qubit_cals is None:
            raise RuntimeError("Mitigator not calibrated. Call cals_from_system() first.")
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
    # Correction — dispatches to the configured method
    # ─────────────────────────────────────────────────────────────────────

    def apply_correction(self, counts, qubits: list[int], **kwargs):
        """Apply REM correction to raw measurement counts.

        Args:
            counts (dict): Qiskit counts dict (little-endian bitstrings →
                shot count).
            qubits (list[int]): Physical qubits corresponding to the bitstring
                bits, in the same order as the bits in the bitstring.
            **kwargs: Additional parameters passed to the underlying method.

                For method='m3':

                - ``distance`` (int | None): max Hamming distance, default
                  ``min(n, 3)``.
                - ``solver`` (str): ``'auto'``, ``'direct'``, or
                  ``'iterative'``.
                - ``max_iter`` (int): max GMRES iterations, default 25.
                - ``tol`` (float): GMRES tolerance, default 1e-4.
                - ``details`` (bool): also return solver metrics.

        Returns:
            dict: Corrected counts (same format as input) for
                ``method='matrix'``, or ``QuasiDistribution`` (mthree) for
                ``method='m3'``. Convert the latter with
                ``quasi.nearest_probability_distribution()``.

        Raises:
            RuntimeError: If calibration has not been loaded, or if a
                requested qubit is not calibrated.
        """
        if self.single_qubit_cals is None:
            raise RuntimeError("Mitigator not calibrated. Call cals_from_system() first.")

        missing = [q for q in qubits if self.single_qubit_cals[q] is None]
        if missing:
            raise RuntimeError(
                f"Uncalibrated qubits: {missing}. "
                "Call cals_from_system(qubits=...) with these qubits first."
            )

        bad = set(qubits) & set(self.faulty_qubits)
        if bad:
            warnings.warn(f"Faulty qubits used in correction: {sorted(bad)}")

        if self.method == "matrix":
            return self._apply_matrix(counts, qubits)
        else:
            return self._apply_m3(counts, qubits, **kwargs)

    # ─────────────────────────────────────────────────────────────────────
    # Matrix method (mitiq)
    # ─────────────────────────────────────────────────────────────────────

    def _apply_matrix(self, counts: dict, qubits: list[int]) -> dict:
        """Apply REM via inverse confusion matrix (mitiq).

        Builds A = ⊗ A_i then applies A⁺ to the counts.
        Memory cost: O(4ⁿ) — practical up to ~12 qubits.
        """
        MeasurementResult, mitigate_measurements = _require_mitiq()

        n = len(qubits)
        if n > 15:
            warnings.warn(
                f"method='matrix' with {n} qubits: the {2**n}×{2**n} matrix "
                "may exhaust memory. Consider using method='m3'."
            )

        # Global inverse confusion matrix (tensor product of pseudo-inverses)
        inv_confusion = self._build_inv_confusion_matrix(qubits)

        # Convert counts → mitiq MeasurementResult (big-endian)
        bitstrings = []
        for bitstring, count in counts.items():
            bits = [int(b) for b in reversed(bitstring)]
            bitstrings.extend([bits] * count)
        noisy_result = MeasurementResult(np.array(bitstrings, dtype=int))

        mitigated_result = mitigate_measurements(noisy_result, inv_confusion)

        # Convert back to Qiskit counts (little-endian)
        corrected_counts: dict = {}
        for bits in mitigated_result.result:
            bs = "".join(str(b) for b in reversed(bits))
            corrected_counts[bs] = corrected_counts.get(bs, 0) + 1

        return corrected_counts

    def _build_inv_confusion_matrix(self, qubits: list[int]) -> np.ndarray:
        """Build the tensor product of pseudo-inverse local calibration matrices.

        A⁺ = pinv(A_q0) ⊗ pinv(A_q1) ⊗ … ⊗ pinv(A_qn-1)
        """
        pinv_matrices = [np.linalg.pinv(self.single_qubit_cals[q]) for q in qubits]
        return reduce(np.kron, pinv_matrices)

    def get_confusion_matrix(self, qubits: list[int]) -> np.ndarray:
        """Return the direct (non-inverted) confusion matrix for the given qubits.

        Useful for visualization (e.g. heatmap).

        Args:
            qubits (list[int]): Physical qubits to include.

        Returns:
            np.ndarray: Tensor product of local calibration matrices.

        Raises:
            RuntimeError: If calibration has not been loaded.
        """
        if self.single_qubit_cals is None:
            raise RuntimeError("Mitigator not calibrated.")
        mats = [self.single_qubit_cals[q] for q in qubits]
        return reduce(np.kron, mats)

    def get_inv_confusion_matrix(self, qubits: list[int]) -> np.ndarray:
        """Return the inverse confusion matrix (tensor product of pseudo-inverses).

        Useful for visualization or reuse in custom correction pipelines.

        Args:
            qubits (list[int]): Physical qubits to include.

        Returns:
            np.ndarray: Tensor product of pseudo-inverse local calibration
                matrices.

        Raises:
            RuntimeError: If calibration has not been loaded.
        """
        if self.single_qubit_cals is None:
            raise RuntimeError("Mitigator not calibrated.")
        return self._build_inv_confusion_matrix(qubits)

    # ─────────────────────────────────────────────────────────────────────
    # M3 method (mthree)
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
        """Apply REM via M3 (Matrix-free Measurement Mitigation, mthree).

        Works on the reduced sub-matrix of observed bitstrings only.
        Scalable to 24 qubits and beyond.

        Args:
            counts (dict): Raw measurement counts.
            qubits (list[int]): Physical qubits corresponding to the bitstring
                bits.
            distance (int | None): Max Hamming distance. ``None`` →
                ``min(n, 3)``. ``-1`` → full distance.
            solver (str): ``'auto'`` selects based on available memory,
                ``'direct'``, or ``'iterative'``.
            max_iter (int): Max GMRES iterations (iterative solver only).
                Default: 25.
            tol (float): GMRES convergence tolerance. Default: 1e-4.
            details (bool): If True, return solver metrics alongside the
                result.
            return_mitigation_overhead (bool): If True, compute and attach the
                mitigation overhead (gamma²).

        Returns:
            QuasiDistribution: Mitigated quasi-distribution, or
                ``(QuasiDistribution, dict)`` if ``details=True``.
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
                f"Bitstring length ({bitstring_len}) != len(qubits) ({num_bits})."
            )

        # Auto-select solver based on available memory
        if solver == "auto":
            free_gb = psutil.virtual_memory().available / 1024**3
            if num_elems <= self.iter_threshold and (
                (num_elems**2 + num_elems) * 8 / 1024**3 < free_gb / 2
            ):
                solver = "direct"
            else:
                solver = "iterative"
            logger.debug("M3 solver auto-selected: %s", solver)

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
            raise ValueError(f"Invalid solver: {solver!r}. Choose 'auto', 'direct', or 'iterative'.")

    # ─────────────────────────────────────────────────────────────────────
    # Internal interface used by mthree solvers
    # ─────────────────────────────────────────────────────────────────────

    def _form_cals(self, qubits) -> np.ndarray:
        """Return a 1-D calibration array in the format expected by mthree solvers.

        Format: [P00_q(n-1), P10_q(n-1), P01_q(n-1), P11_q(n-1), …]
        Qubits are reversed to match M3's internal indexing convention.
        """
        qubits = np.asarray(qubits, dtype=int)
        cals = np.zeros(4 * len(qubits), dtype=np.float32)
        for kk, qubit in enumerate(qubits[::-1]):
            cals[4 * kk: 4 * kk + 4] = self.single_qubit_cals[qubit].astype(np.float32).ravel()
        return cals

    def reduced_cal_matrix(self, counts, qubits, distance=None):
        """Return the reduced calibration sub-matrix for the observed bitstrings (M3).

        Args:
            counts (dict): Measurement counts.
            qubits (list[int]): Physical qubits.
            distance (int | None): Max Hamming distance. ``None`` →
                ``min(n, 3)``.

        Returns:
            tuple[np.ndarray, dict]: Reduced calibration matrix and bitstring
                index mapping.
        """
        _direct_solve, _cal_matrix, *_ = _require_mthree()
        return _cal_matrix(self, counts, qubits, distance)


# ─────────────────────────────────────────────────────────────────────────
# Internal utility
# ─────────────────────────────────────────────────────────────────────────

def _faulty_qubit_checker(cals: list) -> list:
    """Return indices of qubits with inverted calibration (P(0|1) >= P(0|0)).

    A qubit is considered faulty when the probability of reading 0 given
    the qubit was prepared in |1⟩ is at least as high as reading 0 given
    it was prepared in |0⟩ — indicating the readout is unreliable.
    """
    return [
        idx for idx, cal in enumerate(cals)
        if cal is not None and cal[0, 1] >= cal[0, 0]
    ]
