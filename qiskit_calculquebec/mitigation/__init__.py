"""
qiskit_calculquebec.mitigation
================================
Error mitigation techniques for MonarQ.

Available classes
-----------------
ReadoutMitigation
    Readout Error Mitigation (REM).
    Two modes:
      - ``method='matrix'`` : inverse confusion matrix (mitiq), exact but
                              limited to ~12 qubits (2ⁿ × 2ⁿ memory).
      - ``method='m3'``     : M3 (mthree), scalable to 24 qubits and beyond.

ZNEMitigation
    Zero-Noise Extrapolation via mitiq.

DDDMitigation
    Digital Dynamical Decoupling via mitiq.

PauliTwirlingMitigation
    Pauli Twirling (± ZNE) via mitiq.

Installing optional dependencies
---------------------------------
    pip install qiskit-calculquebec[mitigation]
"""

import warnings as _warnings

def _check_optional_deps():
    missing = []
    try:
        import mitiq  # noqa: F401
    except ImportError:
        missing.append("mitiq  (required for ZNEMitigation, DDDMitigation, PauliTwirlingMitigation, ReadoutMitigation(method='matrix'))")
    try:
        import mthree  # noqa: F401
    except ImportError:
        missing.append("mthree (required for ReadoutMitigation(method='m3'))")
    try:
        import psutil  # noqa: F401
    except ImportError:
        missing.append("psutil (required for ReadoutMitigation(method='m3'))")

    if missing:
        _warnings.warn(
            "Missing optional dependencies for qiskit-calculquebec[mitigation]:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\nInstall them with: pip install qiskit-calculquebec[mitigation]",
            UserWarning,
            stacklevel=2,
        )

_check_optional_deps()

from qiskit_calculquebec.mitigation.readout import ReadoutMitigation
from qiskit_calculquebec.mitigation.zne import ZNEMitigation
from qiskit_calculquebec.mitigation.ddd import DDDMitigation
from qiskit_calculquebec.mitigation.pauli_twirling import PauliTwirlingMitigation

__all__ = [
    "ReadoutMitigation",
    "ZNEMitigation",
    "DDDMitigation",
    "PauliTwirlingMitigation",
]
