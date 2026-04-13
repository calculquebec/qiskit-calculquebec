"""
qiskit_calculquebec.mitigation
================================
Techniques de mitigation d'erreurs pour MonarQ.

Classes disponibles
-------------------
ReadoutMitigation
    Mitigation des erreurs de lecture (REM).
    Deux modes :
      - ``method='matrix'``  : matrice de confusion inverse (mitiq), exacte mais
                               limitée à ~12 qubits (mémoire 2ⁿ × 2ⁿ).
      - ``method='m3'``      : M3 (mthree), scalable jusqu'à 24 qubits et au-delà.

ZNEMitigation
    Zero-Noise Extrapolation via mitiq.

DDDMitigation
    Digital Dynamical Decoupling via mitiq.

PauliTwirlingMitigation
    Pauli Twirling (± ZNE) via mitiq.

Installation des dépendances optionnelles
-----------------------------------------
    pip install qiskit-calculquebec[mitigation]
"""

import warnings as _warnings

def _check_optional_deps():
    missing = []
    try:
        import mitiq  # noqa: F401
    except ImportError:
        missing.append("mitiq  (requis pour ZNEMitigation, DDDMitigation, PauliTwirlingMitigation, ReadoutMitigation(method='matrix'))")
    try:
        import mthree  # noqa: F401
    except ImportError:
        missing.append("mthree (requis pour ReadoutMitigation(method='m3'))")
    try:
        import psutil  # noqa: F401
    except ImportError:
        missing.append("psutil (requis pour ReadoutMitigation(method='m3'))")

    if missing:
        _warnings.warn(
            "Dépendances optionnelles manquantes pour qiskit-calculquebec[mitigation] :\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\nInstallez-les avec : pip install qiskit-calculquebec[mitigation]",
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
