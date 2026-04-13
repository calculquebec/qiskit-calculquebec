"""
Qiskit job wrappers for the MonarQ/Yukon backend.

``MonarQJob`` manages a single-circuit job. ``MultiMonarQJob`` sequences
multiple single-circuit jobs and aggregates their results into one
Qiskit ``Result`` object.
"""

import time
from qiskit.providers import JobV1 as Job
from qiskit.providers import JobError, JobTimeoutError
from qiskit.providers.jobstatus import JobStatus
from qiskit.result import Result
from qiskit_calculquebec.API.job import Job as CQJob
from qiskit_calculquebec.API.adapter import ApiAdapter


class MonarQJob(Job):
    """
    Qiskit job wrapper for a single circuit submitted to MonarQ/Yukon.

    If no ``job_id`` is provided, the circuit is submitted to the API
    immediately on construction.

    Parameters
    ----------
    backend : MonarQBackend
        The backend this job was submitted to.
    job_id : str | None
        Existing job ID to track. If ``None``, the circuit is submitted
        immediately and the returned ID is stored.
    circuits : list[QuantumCircuit] | None
        List containing exactly one circuit. Required when ``job_id`` is
        ``None``.
    shots : int
        Number of shots. Default: 1000.
    """

    def __init__(self, backend, job_id=None, circuits=None, shots=1000):
        super().__init__(backend, job_id)
        self._backend = backend
        self.circuits = circuits or []
        self.shots = shots

        if job_id is None:
            self._job_id = self._submit_circuit()
        else:
            self._job_id = job_id

    def _submit_circuit(self) -> str:
        """
        Submit the single circuit to the API and return the job ID.

        Returns
        -------
        str
            Job ID assigned by the scheduler.

        Raises
        ------
        ValueError
            If ``circuits`` does not contain exactly one circuit.
        """
        if not self.circuits or len(self.circuits) != 1:
            raise ValueError("MonarQJob can only submit one circuit at a time.")

        return CQJob(self.circuits[0], self.shots).run_getID()

    def _wait_for_result(self, timeout=None, wait=5) -> dict:
        """
        Poll the API until the job completes or fails.

        Parameters
        ----------
        timeout : float | None
            Maximum number of seconds to wait. ``None`` means no timeout.
        wait : float
            Seconds to sleep between polling attempts. Default: 5.

        Returns
        -------
        dict
            Full API response JSON for the completed job.

        Raises
        ------
        JobTimeoutError
            If ``timeout`` is exceeded before the job completes.
        JobError
            If the job status is ``"FAILED"``.
        """
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if timeout and elapsed >= timeout:
                raise JobTimeoutError("Timed out waiting for result.")

            response = ApiAdapter.job_by_id(self._job_id)
            result = response.json()
            status = result["job"]["status"]["type"]

            if status == "SUCCEEDED":
                break
            elif status == "FAILED":
                raise JobError("Job execution failed.")

            time.sleep(wait)

        return result

    def result(self, timeout=None, wait=5) -> Result:
        """
        Block until the job completes and return a Qiskit ``Result``.

        The API histogram (bitstring → count) is converted to both ``counts``
        and ``memory`` (hex strings repeated according to counts) for full
        Qiskit compatibility.

        Parameters
        ----------
        timeout : float | None
            Maximum seconds to wait. ``None`` means no timeout.
        wait : float
            Seconds between polling attempts. Default: 5.

        Returns
        -------
        Result
            Qiskit result containing ``counts`` and ``memory``.
        """
        job_info = self._wait_for_result(timeout, wait)
        histogram = job_info["result"]["histogram"]

        memory = []
        for bitstring, count in histogram.items():
            val = int(bitstring, 2)
            hex_str = format(val, f"0{((len(bitstring) + 3) // 4)}x")
            memory.extend([hex_str] * count)

        qiskit_results = [
            {
                "success": True,
                "shots": sum(histogram.values()),
                "data": {"counts": histogram, "memory": memory},
            }
        ]

        return Result.from_dict(
            {
                "results": qiskit_results,
                "backend_name": getattr(self._backend, "name", "unknown"),
                "backend_version": getattr(self._backend, "backend_version", "0.0.0"),
                "job_id": self._job_id,
                "success": True,
            }
        )

    def status(self) -> JobStatus:
        """
        Return the current Qiskit ``JobStatus`` for this job.

        Returns
        -------
        JobStatus
            One of ``RUNNING``, ``DONE``, ``QUEUED``, ``CANCELLED``,
            or ``ERROR``.
        """
        response = ApiAdapter.job_by_id(self._job_id)
        status_str = response.json()["job"]["status"]["type"]

        mapping = {
            "RUNNING": JobStatus.RUNNING,
            "SUCCEEDED": JobStatus.DONE,
            "QUEUED": JobStatus.QUEUED,
            "CANCELLED": JobStatus.CANCELLED,
        }
        return mapping.get(status_str, JobStatus.ERROR)

    def submit(self) -> Result:
        """Alias for :meth:`result`. Triggers result retrieval."""
        return self.result()


class MultiMonarQJob(Job):
    """
    Qiskit job wrapper that sequences multiple circuits on a single-job backend.

    MonarQ/Yukon only supports one circuit per API job. This class submits
    each circuit as a separate ``MonarQJob`` and aggregates the results into
    a single Qiskit ``Result`` object.

    Parameters
    ----------
    backend : MonarQBackend
        The backend this job was submitted to.
    circuits : list[QuantumCircuit]
        Circuits to execute sequentially.
    job_id : str | None
        Optional composite job ID. Default: ``"multi_job"``.
    shots : int | None
        Shots per circuit. Falls back to ``backend.options.shots`` if ``None``.
    """

    def __init__(self, backend, circuits, job_id=None, shots=None):
        super().__init__(backend, job_id or "multi_job")
        self._backend = backend
        self.circuits = circuits
        self.shots = shots or getattr(backend.options, "shots", 1000)

        self._individual_jobs = [
            MonarQJob(backend, circuits=[c], shots=self.shots) for c in circuits
        ]

    def _wait_for_result(self, timeout=None, wait=5) -> bool:
        """
        Wait for all individual jobs to complete.

        Parameters
        ----------
        timeout : float | None
            Maximum seconds to wait per job. ``None`` means no timeout.
        wait : float
            Seconds between polling attempts. Default: 5.

        Returns
        -------
        bool
            Always ``True`` when all jobs have completed successfully.
        """
        for job in self._individual_jobs:
            job._wait_for_result(timeout=timeout, wait=wait)
        return True

    def result(self, timeout=None, wait=5) -> Result:
        """
        Collect results from all individual jobs and combine them.

        Parameters
        ----------
        timeout : float | None
            Maximum seconds to wait per job. ``None`` means no timeout.
        wait : float
            Seconds between polling attempts. Default: 5.

        Returns
        -------
        Result
            Combined Qiskit ``Result`` containing one entry per circuit.
        """
        all_results = []

        for job in self._individual_jobs:
            res = job.result(timeout=timeout, wait=wait)

            for exp in res.results:
                data = exp.data
                # Ensure Estimator compatibility: add 'evs' if missing
                if not hasattr(data, "evs") and not hasattr(data, "counts"):
                    ev = getattr(data, "expectation_value", None)
                    setattr(data, "evs", ev if ev is not None else [])
                all_results.append(exp)

        return Result.from_dict(
            {
                "results": [exp.to_dict() for exp in all_results],
                "backend_name": getattr(self._backend, "name", "unknown"),
                "backend_version": getattr(self._backend, "backend_version", "0.0.0"),
                "job_id": self._job_id,
                "success": all(exp.success for exp in all_results),
            }
        )

    def status(self) -> JobStatus:
        """
        Return the aggregate ``JobStatus`` across all individual jobs.

        Returns ``DONE`` only when all jobs have succeeded; returns ``RUNNING``
        if any job is still running; returns ``ERROR`` if any job has failed.

        Returns
        -------
        JobStatus
            Aggregated status.
        """
        statuses = [job.status() for job in self._individual_jobs]

        if all(s == JobStatus.DONE for s in statuses):
            return JobStatus.DONE
        elif any(s == JobStatus.RUNNING for s in statuses):
            return JobStatus.RUNNING
        elif all(s == JobStatus.QUEUED for s in statuses):
            return JobStatus.QUEUED
        elif any(s == JobStatus.ERROR for s in statuses):
            return JobStatus.ERROR
        else:
            return JobStatus.INITIALIZING

    def submit(self) -> Result:
        """Alias for :meth:`result`. Triggers combined result retrieval."""
        return self.result()
