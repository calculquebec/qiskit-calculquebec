"""
Contains a wrapper around the job creation and executing process for MonarQ
"""

from qiskit import QuantumCircuit
import json
import time
from qiskit_calculquebec.API.adapter import ApiAdapter
from qiskit_calculquebec.API.api_utility import ApiUtility


class JobException(Exception):
    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message

    def __repr__(self):
        return self.message


class Job:
    """A wrapper around Thunderhead's jobs operations.
    - converts your circuit to an http request
    - posts a job on monarq
    - periodically checks if the job is done
    - returns results when it's done

    Args :
        - circuit (QuantumCircuit) : the circuit you want to execute
        - circuit_name (str) : the name of the circuit
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        shots=1,
    ):
        self.circuit_dict = ApiUtility.convert_circuit(circuit)
        self.shots = shots

    def run_getID(self) -> str:
        try:
            response = ApiAdapter.post_job(self.circuit_dict, self.shots)
        except:
            raise
        if response.status_code != 200:
            self.raise_api_error(response)

        return json.loads(response.text)["job"]["id"]

    def run(self, max_tries: int = -1) -> dict:
        """
        converts a quantum circuit into a dictionary, readable by thunderhead
        creates a job on thunderhead
        fetches the result until the job is successfull, and returns the result
        """

        if max_tries == -1:
            max_tries = 2**15
        response = None
        try:
            response = ApiAdapter.post_job(self.circuit_dict, self.shots)
        except:
            raise
        if response.status_code == 200:
            current_status = ""
            job_id = json.loads(response.text)["job"]["id"]
            for _ in range(max_tries):
                time.sleep(0.2)
                response = ApiAdapter.job_by_id(job_id)

                if response.status_code != 200:
                    self.raise_api_error(response)

                content = json.loads(response.text)
                status = content["job"]["status"]["type"]
                if current_status != status:
                    current_status = status

                if status != "SUCCEEDED":
                    continue

                return content["result"]["histogram"]
            raise JobException(
                "Couldn't finish job. Stuck on status : " + str(current_status)
            )
        else:
            self.raise_api_error(response)

    def raise_api_error(self, response):
        """
        this raises an error by parsing the json body of the response, and using the response text as message
        """
        error = json.loads(response.text)

        raise JobException(
            f"API ERROR: {error.get('code')}, {error.get('error')}, {error}"
        )
