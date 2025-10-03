import time
from qiskit.providers import JobV1 as Job
from qiskit.providers import JobError, JobTimeoutError
from qiskit.providers.jobstatus import JobStatus
from qiskit.result import Result
from qiskit_calculquebec.API.job import Job as CQJob
from qiskit_calculquebec.API.adapter import ApiAdapter


class AnyonJob(Job):
    """
    A wrapper for submitting and managing a single circuit to a Yukon/Anyon backend.

    Handles:
    - Circuit submission
    - Polling for job completion
    - Conversion of API results to Qiskit's Result object
    """
    
    def __init__(self, backend, job_id=None, circuits=None, shots=1000):
        super().__init__(backend, job_id)
        self._backend = backend
        self.circuits = circuits or []
        self.shots = shots

        # If no job ID is provided, submit immediately
        if job_id is None:
            self._job_id = self._submit_circuit()
        else:
            self._job_id = job_id

    def _submit_circuit(self):
        """Submit a single circuit to the API and return the job ID."""
        if not self.circuits or len(self.circuits) != 1:
            raise ValueError("AnyonJob can only submit one circuit at a time")

        circuit = self.circuits[0]
        result = CQJob(circuit, self.shots).run_getID()
        return result

    def _wait_for_result(self, timeout=None, wait=5):
        """
        Poll the API until the job completes or fails.

        Args:
            timeout: Maximum time (seconds) to wait
            wait: Delay (seconds) between polling

        Returns:
            dict: API response JSON for the completed job
        """
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if timeout and elapsed >= timeout:
                raise JobTimeoutError('Timed out waiting for result')

            response = ApiAdapter.job_by_id(self._job_id)
            result = response.json()
            status = result["job"]["status"]["type"]

            if status == "SUCCEEDED":
                break
            elif status == "FAILED":
                raise JobError('Job execution failed')

            time.sleep(wait)

        return result

    def result(self, timeout=None, wait=5):
        """
        Return a Qiskit Result object from the completed job.
        Converts API histogram into counts and memory format.
        """
        job_info = self._wait_for_result(timeout, wait)
        histogram = job_info['result']['histogram']

        # Build memory as hex strings repeated according to counts
        memory = []
        for bitstring, count in histogram.items():
            val = int(bitstring, 2)  # binary -> int
            hex_str = format(val, f'0{((len(bitstring)+3)//4)}x')  # int -> hex with proper padding
            memory.extend([hex_str] * count)

        # Qiskit expects both counts and memory
        qiskit_results = [{
            'success': True,
            'shots': sum(histogram.values()),
            'data': {
                'counts': histogram,
                'memory': memory
            }
        }]

        return Result.from_dict({
            'results': qiskit_results,
            'backend_name': getattr(self._backend, 'name', 'unknown'),
            'backend_version': getattr(self._backend, 'backend_version', '0.0.0'),
            'job_id': self._job_id,
            'success': True,
        })

    def status(self):
        """Map the backend job status to Qiskit JobStatus."""
        response = ApiAdapter.job_by_id(self._job_id)
        status_str = response.json()["job"]["status"]["type"]

        mapping = {
            "RUNNING": JobStatus.RUNNING,
            "SUCCEEDED": JobStatus.DONE,
            "QUEUED": JobStatus.QUEUED,
            "CANCELLED": JobStatus.CANCELLED
        }

        return mapping.get(status_str, JobStatus.ERROR)

    def submit(self):
        """Convenience method to trigger result retrieval."""
        return self.result()


class MultiAnyonJob(Job):
    """
    Wrapper to handle multiple circuits sequentially on a backend
    that only supports one circuit per job.

    This class manages multiple AnyonJob instances internally.
    """

    def __init__(self, backend, circuits, job_id=None, shots=None):
        super().__init__(backend, job_id or "multi_job")
        self._backend = backend
        self.circuits = circuits
        self.shots = shots or getattr(backend.options, "shots", 1000)

        # Create individual AnyonJobs for each circuit
        self._individual_jobs = [
            AnyonJob(backend, circuits=[c], shots=self.shots)
            for c in circuits
        ]

    def _wait_for_result(self, timeout=None, wait=5):
        """Wait for all individual jobs to complete."""
        for job in self._individual_jobs:
            job._wait_for_result(timeout=timeout, wait=wait)
        return True

    def result(self, timeout=None, wait=5):
        """
        Combine results from all individual AnyonJobs into a single Qiskit Result.
        Ensures compatibility with Estimator by setting 'evs' if missing.
        """
        all_results = []

        for job in self._individual_jobs:
            res = job.result(timeout=timeout, wait=wait)

            for exp in res.results:
                data = exp.data

                # Ensure compatibility: add 'evs' if missing
                if not hasattr(data, 'evs') and not hasattr(data, 'counts'):
                    ev = getattr(data, 'expectation_value', None)
                    setattr(data, 'evs', ev if ev is not None else [])

                all_results.append(exp)

        return Result.from_dict({
            "results": [exp.to_dict() for exp in all_results],
            "backend_name": getattr(self._backend, "name", "unknown"),
            "backend_version": getattr(self._backend, "backend_version", "0.0.0"),
            "job_id": self._job_id,
            "success": all(exp.success for exp in all_results)
        })

    def status(self):
        """Aggregate status from all individual jobs."""
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

    def submit(self):
        """Convenience method to trigger combined result retrieval."""
        return self.result()
