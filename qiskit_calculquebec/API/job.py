"""
Low-level job submission and result polling for MonarQ.

Converts a ``QuantumCircuit`` to the Thunderhead wire format, posts the job,
polls until completion, and returns the result histogram.
"""

from qiskit import QuantumCircuit
import json
import time
from qiskit_calculquebec.API.adapter import ApiAdapter
from qiskit_calculquebec.API.api_utility import ApiUtility


class JobException(Exception):
    """
    Raised when a job submission or polling operation fails.

    Parameters
    ----------
    message : str
        Human-readable description of the failure.
    """

    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message

    def __repr__(self):
        return self.message


class Job:
    """
    Wrapper for submitting a single circuit to the MonarQ/Thunderhead API.

    Converts the circuit to dictionary format, posts the job, polls for
    completion, and returns the result histogram.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to execute.
    shots : int
        Number of shots. Default: 1.
    """

    def __init__(self, circuit: QuantumCircuit, shots: int = 1):
        self.circuit_dict = ApiUtility.convert_circuit(circuit)
        self.shots = shots

    def run_getID(self) -> str:
        """
        Submit the job and return its ID without waiting for completion.

        Returns
        -------
        str
            The job ID assigned by the scheduler.

        Raises
        ------
        JobException
            If submission fails or the response is not 200.
        """
        response = ApiAdapter.post_job(self.circuit_dict, self.shots)
        if response.status_code != 200:
            self.raise_api_error(response)
        return json.loads(response.text)["job"]["id"]

    def run(self, max_tries: int = -1) -> dict:
        """
        Submit the job and block until it succeeds, then return the histogram.

        Polls the API every 200 ms. The job is considered done when its status
        is ``"SUCCEEDED"``.

        Parameters
        ----------
        max_tries : int
            Maximum number of polling iterations. ``-1`` means effectively
            unlimited (2¹⁵ attempts). Default: -1.

        Returns
        -------
        dict
            Result histogram mapping bitstrings to shot counts.

        Raises
        ------
        JobException
            If the job does not complete within ``max_tries`` iterations, or
            if submission fails.
        """
        if max_tries == -1:
            max_tries = 2**15

        response = ApiAdapter.post_job(self.circuit_dict, self.shots)
        if response.status_code != 200:
            self.raise_api_error(response)

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

            if status == "SUCCEEDED":
                return content["result"]["histogram"]

        raise JobException(
            f"Job did not complete. Last status: {current_status}"
        )

    def raise_api_error(self, response):
        """
        Parse an API error response and raise a ``JobException``.

        Parameters
        ----------
        response : requests.Response
            The failed HTTP response.

        Raises
        ------
        JobException
            Always raised with the parsed error code and message.
        """
        error = json.loads(response.text)
        raise JobException(
            f"API ERROR: {error.get('code')}, {error.get('error')}, {error}"
        )
