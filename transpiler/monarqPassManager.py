from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import BasisTranslator, UnrollCustomDefinitions

from .equivalenceRules import getEquivalenceRules

def getPassManager(target):

    pm = PassManager()
    custom_lib = getEquivalenceRules()    

    # ajouter les définitions des portes natives de monarq
    pm.append(UnrollCustomDefinitions(custom_lib, target.operation_names))

    # convertir vers l'ensemble de porte natives
    pm.append(BasisTranslator(custom_lib, target.operation_names))

    # TODO ajouter d'autres étapes d'optimisation au besoin

    return pm
