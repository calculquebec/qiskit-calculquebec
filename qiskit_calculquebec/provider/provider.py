from qiskit.providers.providerutils import filter_backends
from qiskit_calculquebec.backends.monarq_backend import MonarQBackend as MyBackend


class CalculQuebecProvider:
    """
    Provider for Calcul Québec quantum backends.

    This class manages available backends and provides access by name or filters.
    """

    def __init__(self, token=None):
        """
        Initialize the provider.

        Args:
            token (str, optional): API token for authentication.
        """
        self.token = token
        # Initialize the available backends; pass the provider reference to each
        self._backends = [MyBackend(provider=self)]

    def get_backend(self, name):
        """
        Return a backend matching the given name.

        Args:
            name (str): Name of the backend.

        Returns:
            Backend: Matching backend instance.

        Raises:
            ValueError: If no backend matches the given name.
        """
        for backend in self._backends:
            # backend.name is a property, not a method
            if backend.name == name:
                return backend
        raise ValueError(f"No backend found with name: {name}")

    def backends(self, name=None, filters=None, **kwargs):
        """
        Return a list of available backends, optionally filtered by name or custom filters.

        Args:
            name (str, optional): Only return backends matching this name.
            filters (callable, optional): Custom filter function for backends.
            **kwargs: Additional arguments passed to filter_backends.

        Returns:
            List[Backend]: List of matching backends.
        """
        backends = self._backends

        # Filter by exact name if provided
        if name is not None:
            backends = [backend for backend in backends if backend.name == name]

        # Apply any additional filters using Qiskit's utility
        return list(filter_backends(backends, filters=filters, **kwargs))
