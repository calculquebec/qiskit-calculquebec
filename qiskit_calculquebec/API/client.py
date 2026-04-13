"""
API client classes for Calcul Québec quantum backends.

``CalculQuebecClient`` is the primary client for connecting to MonarQ/Yukon.
``MonarqClient`` is a deprecated alias kept for backwards compatibility.
"""


class ProjectParameterError(ValueError):
    """
    Raised when project parameter validation fails in a client constructor.

    Either ``project_name`` or ``project_id`` must be provided, but not both.
    """
    pass


class ApiClient:
    """
    Data container holding authentication and routing information for the API.

    Either ``project_name`` or ``project_id`` must be supplied. If both are
    provided, ``project_id`` takes precedence and ``project_name`` is ignored.

    Parameters
    ----------
    host : str
        Base URL of the Thunderhead server (e.g. ``"https://..."``).
    user : str
        User identifier (username).
    access_token : str
        API access token for authentication.
    realm : str
        Organizational realm associated with the machine.
    project_name : str
        Name of the project. Resolved to a project ID at initialization.
        Mutually exclusive with ``project_id``.
    project_id : str
        Direct project ID. Mutually exclusive with ``project_name``.
    circuit_name : str
        Label attached to submitted circuits. Default: ``"none"``.
    """

    @property
    def project_name(self) -> str:
        """Project name used for ID resolution."""
        return self._project_name

    @property
    def project_id(self) -> str:
        """Project ID used in job submission requests."""
        return self._project_id

    @project_id.setter
    def project_id(self, value: str):
        """Set the project ID."""
        self._project_id = value

    @property
    def circuit_name(self) -> str:
        """Circuit label attached to submitted jobs."""
        return self._circuit_name

    @circuit_name.setter
    def circuit_name(self, value: str):
        """Set the circuit name."""
        self._circuit_name = value

    @property
    def machine_name(self) -> str:
        """Name of the target quantum machine."""
        return self._machine_name

    @machine_name.setter
    def machine_name(self, value: str):
        """Set the machine name."""
        self._machine_name = value

    def __init__(
        self,
        host: str,
        user: str,
        access_token: str,
        realm: str,
        project_name: str = "",
        project_id: str = "",
        circuit_name: str = "none",
    ):
        if project_name == "" and project_id == "":
            raise ProjectParameterError(
                "Either project_name or project_id must be provided."
            )
        if project_name != "" and project_id != "":
            # project_id takes precedence when both are given
            project_name = ""

        self.host = host
        self.user = user
        self.access_token = access_token
        self.realm = realm
        self._machine_name = ""
        self._project_name = project_name
        self._project_id = project_id
        self._circuit_name = circuit_name


class CalculQuebecClient(ApiClient):
    """
    Client for Calcul Québec quantum infrastructure.

    Specializes ``ApiClient`` with the ``"calculqc"`` realm and a simplified
    constructor signature (no ``realm`` argument required).

    Parameters
    ----------
    host : str
        Base URL of the Thunderhead server.
    user : str
        User identifier.
    access_token : str
        API access token.
    project_name : str
        Project name (mutually exclusive with ``project_id``).
    project_id : str
        Project ID (mutually exclusive with ``project_name``).
    circuit_name : str
        Label for submitted circuits. Default: ``"none"``.
    """

    def __init__(
        self,
        host: str,
        user: str,
        access_token: str,
        project_name: str = "",
        project_id: str = "",
        circuit_name: str = "none",
    ):
        super().__init__(
            host,
            user,
            access_token,
            "calculqc",
            project_name,
            project_id,
            circuit_name,
        )


class MonarqClient(CalculQuebecClient):
    """
    Deprecated alias for ``CalculQuebecClient``.

    .. deprecated::
        Use ``CalculQuebecClient`` instead. This class will be removed in a
        future release.

    Parameters
    ----------
    host : str
        Base URL of the Thunderhead server.
    user : str
        User identifier.
    access_token : str
        API access token.
    project_name : str
        Project name (mutually exclusive with ``project_id``).
    project_id : str
        Project ID (mutually exclusive with ``project_name``).
    circuit_name : str
        Label for submitted circuits. Default: ``"none"``.
    """

    def __init__(
        self,
        host: str,
        user: str,
        access_token: str,
        project_name: str = "",
        project_id: str = "",
        circuit_name: str = "none",
    ):
        super().__init__(
            host,
            user,
            access_token,
            project_name,
            project_id,
            circuit_name,
        )
        import warnings
        warnings.warn(
            "MonarqClient is deprecated and will be removed in a future release. "
            "Use CalculQuebecClient instead.",
            DeprecationWarning,
            stacklevel=2,
        )
