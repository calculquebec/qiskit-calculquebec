from qiskit.providers.fake_provider import GenericBackendV2
from API.adapter import ApiAdapter
from API.client import MonarqClient
from API.job import Job
from transpiler.monarqPassManager import getPassManager
from transpiler.monarqTarget import getTarget
from monarq_data import connectivity


class monarq_backend(GenericBackendV2):
    def __init__(self):
        super().__init__(num_qubits=len(connectivity.get("qubits")))   
 
    def run(self, circuit, **kwargs):
        
        # if kwargs == {}:
            #TODO si on veut default vers une simulation quand on a pas de paramètres pour l'API
        # else:
            return self._send_job(circuit, kwargs)
        
        
    def _send_job(self, circuit, kwargs):
        target = getTarget()
        pm = getPassManager(target)
        transpiled_circuit = pm.run(circuit)

        ApiAdapter.initialize(MonarqClient(
            host = kwargs.get("host"),
            user=kwargs.get("user"),
            access_token=kwargs.get("access_token"),
            project_name=kwargs.get("project_name")
        ))
        
        result = Job(transpiled_circuit, kwargs.get("shots"), kwargs.get("circuit_name")).run()
        return result