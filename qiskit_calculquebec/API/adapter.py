"""
API adapter for the Calcul Québec / MonarQ backend.

Provides the ``ApiAdapter`` singleton, which wraps every HTTP call required
to communicate with the Thunderhead job scheduler.
"""

from qiskit_calculquebec.API.api_utility import ApiUtility, routes, keys, queries
import requests
import json
from qiskit_calculquebec.API.client import ApiClient
from datetime import datetime, timedelta
from qiskit_calculquebec.API.retry_decorator import retry


class ApiException(Exception):
    """
    Raised when an API call returns a non-200 HTTP status code.

    Parameters
    ----------
    code : int
        HTTP status code returned by the server.
    message : str
        Human-readable error description.
    """

    def __init__(self, code: int, message: str):
        self.message = f"API ERROR : {code}, {message}"
        super().__init__(self.message)


class ProjectException(Exception):
    """
    Base exception for project-related errors.

    Parameters
    ----------
    message : str
        Human-readable error description.
    """

    def __init__(self, message: str):
        self.message = f"PROJECT ERROR : {message}"
        super().__init__(self.message)


class MultipleProjectsException(ProjectException):
    """
    Raised when multiple projects share the same name.

    Use the project ID instead of the name when this occurs.

    Parameters
    ----------
    projects : list
        List of project dicts (each containing ``name`` and ``id``) that
        matched the requested name.
    """

    def __init__(self, projects: list):
        message = (
            "Multiple projects found with the same name. "
            "Use the project ID instead of the name when creating the client.\n"
            "Projects found:\n"
        )
        for project in projects:
            message += f"  Name: {project[keys.NAME]}, ID: {project[keys.ID]}\n"
        super().__init__(message)


class NoProjectFoundException(ProjectException):
    """
    Raised when no project matches the requested name.

    Parameters
    ----------
    project_name : str
        The name that was searched for.
    """

    def __init__(self, project_name: str):
        super().__init__(f"No project found with name: {project_name}")


class ApiAdapter(object):
    """
    Singleton wrapper around the Thunderhead REST API.

    Initialize with ``ApiAdapter.initialize(client)``, then access the
    instance with ``ApiAdapter.instance()``. Machine, benchmark, and
    qubit/coupler data are cached for up to 24 hours.

    Provides:
    - Job submission and retrieval
    - Machine listing and lookup
    - Benchmark / calibration data retrieval
    - Project ID resolution by name
    """

    _qubits_and_couplers = None
    _machine = None
    _benchmark = None
    _last_update = None

    client: ApiClient
    headers: dict[str, str]
    _instance: "ApiAdapter" = None

    def __init__(self):
        raise Exception(
            "Use ApiAdapter.initialize(client) and ApiAdapter.instance() instead."
        )

    @staticmethod
    def clean_cache():
        """Clear all cached API responses (machine, benchmark, qubits/couplers)."""
        ApiAdapter._qubits_and_couplers = None
        ApiAdapter._machine = None
        ApiAdapter._benchmark = None
        ApiAdapter._last_update = None

    @classmethod
    def instance(cls) -> "ApiAdapter":
        """
        Return the singleton ``ApiAdapter`` instance.

        Returns
        -------
        ApiAdapter
            The initialized singleton, or ``None`` if not yet initialized.
        """
        return cls._instance

    @classmethod
    def initialize(cls, client: ApiClient):
        """
        Create and configure the singleton ``ApiAdapter`` instance.

        If ``client.project_name`` is set, the corresponding project ID is
        resolved automatically via the API.

        Parameters
        ----------
        client : ApiClient
            Authenticated client containing host, credentials, and project info.
        """
        cls._instance = cls.__new__(cls)
        cls._instance.headers = ApiUtility.headers(
            client.user, client.access_token, client.realm
        )
        cls._instance.client = client
        if client.project_name != "":
            cls._instance.client.project_id = ApiAdapter.get_project_id_by_name(
                client.project_name
            )

        cls._qubits_and_couplers: dict = None
        cls._machine: dict = None
        cls._benchmark: dict = None
        cls._last_update: datetime = None

    @staticmethod
    def is_last_update_expired() -> bool:
        """
        Return whether the cached data is older than 24 hours.

        Returns
        -------
        bool
            ``True`` if the cache is stale and should be refreshed.
        """
        return datetime.now() - ApiAdapter._last_update > timedelta(hours=24)

    @staticmethod
    @retry(3)
    def get_project_id_by_name(project_name: str = "default") -> str:
        """
        Resolve a project name to its unique ID.

        Parameters
        ----------
        project_name : str
            Name of the project to look up. Default: ``"default"``.

        Returns
        -------
        str
            The project ID.

        Raises
        ------
        MultipleProjectsException
            If more than one project shares the given name.
        NoProjectFoundException
            If no project matches the given name.
        ApiException
            If the HTTP request fails.
        """
        res = requests.get(
            ApiAdapter.instance().client.host
            + routes.PROJECTS
            + queries.NAME
            + "="
            + project_name,
            headers=ApiAdapter.instance().headers,
        )

        if res.status_code != 200:
            ApiAdapter.raise_exception(res)

        converted = json.loads(res.text)
        projects = converted.get(keys.ITEMS, [])
        matching_projects = [
            p for p in projects if p.get(keys.NAME) == project_name
        ]

        if len(matching_projects) > 1:
            raise MultipleProjectsException(matching_projects)
        if len(matching_projects) == 1:
            return matching_projects[0][keys.ID]

        raise NoProjectFoundException(project_name)

    @staticmethod
    @retry(3)
    def get_machine_by_name(machine_name: str) -> dict:
        """
        Fetch machine metadata by name, caching the result.

        Parameters
        ----------
        machine_name : str
            Name of the machine to retrieve.

        Returns
        -------
        dict
            Machine metadata as returned by the API.

        Raises
        ------
        ApiException
            If the HTTP request fails.
        """
        if ApiAdapter._machine is None:
            route = (
                ApiAdapter.instance().client.host
                + routes.MACHINES
                + queries.MACHINE_NAME
                + "="
                + machine_name
            )
            res = requests.get(route, headers=ApiAdapter.instance().headers)
            if res.status_code != 200:
                ApiAdapter.raise_exception(res)
            ApiAdapter._machine = json.loads(res.text)

        return ApiAdapter._machine

    @staticmethod
    @retry(3)
    def get_qubits_and_couplers(machine_name: str) -> dict:
        """
        Return qubit and coupler fidelity data from the latest benchmark.

        Parameters
        ----------
        machine_name : str
            Name of the machine.

        Returns
        -------
        dict
            ``resultsPerDevice`` section of the benchmark, containing T1, T2,
            single-qubit gate fidelities, CZ fidelities, and readout fidelities.
        """
        benchmark = ApiAdapter.get_benchmark(machine_name)
        return benchmark[keys.RESULTS_PER_DEVICE]

    @staticmethod
    @retry(3)
    def get_benchmark(machine_name: str) -> dict:
        """
        Fetch the latest benchmark for a machine, caching the result for 24 hours.

        Parameters
        ----------
        machine_name : str
            Name of the machine.

        Returns
        -------
        dict
            Full benchmark response from the API.

        Raises
        ------
        ApiException
            If the HTTP request fails.
        """
        if ApiAdapter._benchmark is None or ApiAdapter.is_last_update_expired():
            machine = ApiAdapter.get_machine_by_name(machine_name)
            machine_id = machine[keys.ITEMS][0][keys.ID]

            route = (
                ApiAdapter.instance().client.host
                + routes.MACHINES
                + "/"
                + machine_id
                + routes.BENCHMARKING
            )
            res = requests.get(route, headers=ApiAdapter.instance().headers)
            if res.status_code != 200:
                ApiAdapter.raise_exception(res)
            ApiAdapter._benchmark = json.loads(res.text)
            ApiAdapter._last_update = datetime.now()

        return ApiAdapter._benchmark

    @staticmethod
    @retry(3)
    def post_job(circuit: dict, shot_count: int = 1) -> requests.Response:
        """
        Submit a new job to the scheduler.

        Parameters
        ----------
        circuit : dict
            Circuit in Thunderhead dictionary format.
        shot_count : int
            Number of shots to execute. Default: 1.

        Returns
        -------
        requests.Response
            HTTP response from the ``POST /jobs`` endpoint.

        Raises
        ------
        ApiException
            If the HTTP request fails.
        """
        project_id = ApiAdapter.instance().client.project_id
        circuit_name = ApiAdapter.instance().client.circuit_name
        machine_name = ApiAdapter.instance().client.machine_name
        body = ApiUtility.job_body(
            circuit, circuit_name, project_id, machine_name, shot_count
        )
        res = requests.post(
            ApiAdapter.instance().client.host + routes.JOBS,
            data=json.dumps(body),
            headers=ApiAdapter.instance().headers,
        )
        if res.status_code != 200:
            ApiAdapter.raise_exception(res)
        return res

    @staticmethod
    @retry(3)
    def list_jobs() -> requests.Response:
        """
        Retrieve all jobs for the authenticated user.

        Returns
        -------
        requests.Response
            HTTP response from the ``GET /jobs`` endpoint.

        Raises
        ------
        ApiException
            If the HTTP request fails.
        """
        res = requests.get(
            ApiAdapter.instance().client.host + routes.JOBS,
            headers=ApiAdapter.instance().headers,
        )
        if res.status_code != 200:
            ApiAdapter.raise_exception(res)
        return res

    @staticmethod
    @retry(3)
    def job_by_id(id: str) -> requests.Response:
        """
        Retrieve a specific job by its ID.

        Parameters
        ----------
        id : str
            Job ID to retrieve.

        Returns
        -------
        requests.Response
            HTTP response from the ``GET /jobs/{id}`` endpoint.

        Raises
        ------
        ApiException
            If the HTTP request fails.
        """
        res = requests.get(
            ApiAdapter.instance().client.host + routes.JOBS + f"/{id}",
            headers=ApiAdapter.instance().headers,
        )
        if res.status_code != 200:
            ApiAdapter.raise_exception(res)
        return res

    @staticmethod
    @retry(3)
    def list_machines(online_only: bool = False) -> list[dict]:
        """
        Return the list of available machines.

        Parameters
        ----------
        online_only : bool
            If ``True``, return only machines with status ``"online"``.
            Default: ``False``.

        Returns
        -------
        list[dict]
            List of machine metadata dicts.

        Raises
        ------
        ApiException
            If the HTTP request fails.
        """
        res = requests.get(
            ApiAdapter.instance().client.host + routes.MACHINES,
            headers=ApiAdapter.instance().headers,
        )
        if res.status_code != 200:
            ApiAdapter.raise_exception(res)
        return [
            m
            for m in json.loads(res.text)[keys.ITEMS]
            if not online_only or m[keys.STATUS] == keys.ONLINE
        ]

    def get_connectivity_for_machine(machine_name: str) -> dict:
        """
        Return the coupler-to-qubit connectivity map for a machine.

        Parameters
        ----------
        machine_name : str
            Name of the machine.

        Returns
        -------
        dict
            Coupler-to-qubit mapping describing hardware connectivity.

        Raises
        ------
        ApiException
            If no machine with the given name is found.
        """
        machines = ApiAdapter.list_machines()
        target = [m for m in machines if m[keys.NAME] == machine_name]
        if len(target) < 1:
            raise ApiException(404, f"No machine available with name {machine_name}")

        return target[0][keys.COUPLER_TO_QUBIT_MAP]

    @staticmethod
    def raise_exception(res):
        """
        Parse an HTTP response and raise an ``ApiException``.

        Attempts to extract a human-readable error message from the JSON body.

        Parameters
        ----------
        res : requests.Response
            The failed HTTP response.

        Raises
        ------
        ApiException
            Always raised with the status code and parsed message.
        """
        message = res

        if hasattr(message, "text"):
            message = message.text

        try:
            message = json.loads(message)
            if "error" in message:
                message = message["error"]
        except Exception:
            pass

        raise ApiException(res.status_code, message)
