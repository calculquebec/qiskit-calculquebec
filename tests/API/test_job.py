import pytest
from unittest.mock import patch
from qiskit import QuantumCircuit
from qiskit_calculquebec.API.job import Job, JobException
from qiskit_calculquebec.API.adapter import ApiAdapter
import json
import time


# Mock response for job_by_id
class ResponseJobById:
    def __init__(self, status_code, job_status):
        self.status_code = status_code
        if status_code == 400:
            self.text = '{"code":400,"error":"this is an error"}'
        elif status_code == 200:
            self.text = json.dumps(
                {"job": {"status": {"type": job_status}}, "result": {"histogram": 42}}
            )


# Mock response for errors
class ResponseError:
    def __init__(self):
        self.status_code = 400
        self.text = '{"code":400,"error":"this is an error"}'


# Simple QuantumCircuit substitute for testing
class DummyCircuit(QuantumCircuit):
    pass


@pytest.fixture
def mock_convert_circuit():
    with patch("qiskit_calculquebec.API.api_utility.ApiUtility.convert_circuit") as m:
        m.return_value = {"dummy": "circuit"}
        yield m


@pytest.fixture
def mock_post_job():
    with patch("qiskit_calculquebec.API.adapter.ApiAdapter.post_job") as m:
        yield m


@pytest.fixture
def mock_job_by_id():
    with patch("qiskit_calculquebec.API.adapter.ApiAdapter.job_by_id") as m:
        yield m


class TestJob:
    def test_run_getID_success_and_failure(self, mock_convert_circuit, mock_post_job):
        # Failure response
        mock_post_job.return_value.status_code = 400
        mock_post_job.return_value.text = '{"code":400,"error":"this is an error"}'
        with pytest.raises(Exception):
            Job(DummyCircuit()).run_getID()

        # Success response
        mock_post_job.return_value.status_code = 200
        mock_post_job.return_value.text = '{"job":{"id":"123"}}'
        job_id = Job(DummyCircuit()).run_getID()
        assert job_id == "123"

    def test_run_typical_flow(
        self, mock_convert_circuit, mock_post_job, mock_job_by_id
    ):
        # Setup post_job success
        mock_post_job.return_value.status_code = 200
        mock_post_job.return_value.text = '{"job":{"id":"123"}}'

        # Generator to simulate job status updates
        call_count = {"count": 0}

        def side_effect_job_by_id(job_id):
            call_count["count"] += 1
            if call_count["count"] < 3:
                return ResponseJobById(200, "RUNNING")
            return ResponseJobById(200, "SUCCEEDED")

        mock_job_by_id.side_effect = side_effect_job_by_id

        result = Job(DummyCircuit()).run()
        assert result == 42
        assert call_count["count"] == 3

    def test_run_job_by_id_failure(
        self, mock_convert_circuit, mock_post_job, mock_job_by_id
    ):
        mock_post_job.return_value.status_code = 200
        mock_post_job.return_value.text = '{"job":{"id":"123"}}'

        # Job_by_id returns error
        mock_job_by_id.return_value = ResponseJobById(400, "RUNNING")
        with pytest.raises(JobException):
            Job(DummyCircuit()).run()

    def test_run_iteration_limit(
        self, mock_convert_circuit, mock_post_job, mock_job_by_id
    ):
        mock_post_job.return_value.status_code = 200
        mock_post_job.return_value.text = '{"job":{"id":"123"}}'

        # Job never succeeds
        mock_job_by_id.return_value = ResponseJobById(200, "RUNNING")
        with pytest.raises(JobException):
            Job(DummyCircuit()).run(max_tries=2)

    def test_raise_api_error(self):
        job = Job(DummyCircuit())
        response = ResponseError()
        with pytest.raises(JobException):
            job.raise_api_error(response)
