from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_calculquebec.API.adapter import ApiAdapter
from qiskit_calculquebec.API.client import MonarqClient
from qiskit_calculquebec.API.job import Job
from transpiler.monarqPassManager import getPassManager
from transpiler.transpiler_utility import remove_measurements, append_measurements
from monarq_data import connectivity


class monarq_backend(GenericBackendV2):
    def __init__(self):
        super().__init__(num_qubits=len(connectivity.get("qubits")))

    def run(self, circuit, **kwargs):

        # if kwargs == {}:
        # TODO si on veut default vers une simulation quand on a pas de paramètres pour l'API
        # else:
        return self._send_job(circuit, kwargs)

    def _send_job(self, circuit, kwargs):
        pm = getPassManager()

        # on gère les mesures nous-mêmes
        # le transpilateur n'a pas d'option pour empêcher les mesures intermédiaires (mid-circuit measurement)
        no_measures_circuit, measures = remove_measurements(circuit)
        transpiled_no_measures_circuit = pm.run(no_measures_circuit)
        transpiled_circuit = append_measurements(
            transpiled_no_measures_circuit, measures
        )

        ApiAdapter.initialize(
            MonarqClient(host=kwargs.get("host"),
                user=kwargs.get("user"),
                access_token=kwargs.get("access_token"),
                project_name=kwargs.get("project_name"),
            )
        )

        result = Job(
            transpiled_circuit, kwargs.get("shots"), kwargs.get("circuit_name")
        ).run()
        return result
