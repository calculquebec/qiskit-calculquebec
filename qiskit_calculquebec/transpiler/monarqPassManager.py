from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    UnrollCustomDefinitions,
    BasisTranslator,
    TrivialLayout,
    ApplyLayout,
    LookaheadSwap,
)
from .equivalenceRules import getEquivalenceRules
from qiskit_calculquebec.transpiler.monarqTarget import getTarget


def getPassManager():

    pm = PassManager()
    target = getTarget()
    custom_lib = getEquivalenceRules()
    coupling_map = target.build_coupling_map()

    # 1: associer les qubits du circuit aux qubits de la machine
    # TODO place à amélioration, à modifier en fonction des portes a 2 qubits du circuit
    # possibilité d'inverser l'ordre des étapes 1 et 2 au besoin
    pm.append(TrivialLayout(coupling_map))
    pm.append(ApplyLayout())

    # 2: convertir vers l'ensemble de porte natives
    pm.append(UnrollCustomDefinitions(custom_lib, target.operation_names))
    pm.append(BasisTranslator(custom_lib, target.operation_names))

    # 3: swap les qubits pour que les 2qubit gates respectent la map
    pm.append(LookaheadSwap(coupling_map))

    # 4: re-convertir vers l'ensemble de porte natives (on vient d'ajouter des SWAP)
    # oui, c'est les 2 mêmes lignes que plus haut dans "2:"
    # non, on peut pas just les mettre une fois parce que cette étape risque d'ajouter des portes a 2 qubits sur des qubits non-adjacents
    pm.append(UnrollCustomDefinitions(custom_lib, target.operation_names))
    pm.append(BasisTranslator(custom_lib, target.operation_names))

    # TODO ajouter d'autres étapes au besoin

    return pm
