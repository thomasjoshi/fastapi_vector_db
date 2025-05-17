from typing import Callable, Optional, Protocol
from uuid import UUID

from loguru import logger

from app.domain.models import Library
from app.repos.in_memory import InMemoryRepo, NotFoundError as RepoNotFoundError
from app.services.errors import NotFoundError


class MetricsCallback(Protocol):
    """Protocol for metrics callbacks."""
    def __call__(self, metric_name: str, **kwargs) -> None: ...


def noop_metrics_callback(metric_name: str, **kwargs) -> None:
    """Default no-op metrics callback."""
    pass


class LibraryService:
    """
    Service for managing libraries.
    Uses an InMemoryRepo for storage and provides higher-level operations.
    Includes logging and metrics for observability.
    """

    def __init__(self, repo: InMemoryRepo, metrics_callback: MetricsCallback = noop_metrics_callback) -> None:
        """
        Initialize the library service.
        
        Args:
            repo: The repository to use for storage
            metrics_callback: Optional callback for metrics collection
        """
        self._repo = repo
        self._metrics = metrics_callback

    def _ensure_exists(self, library_id: UUID) -> None:
        """
        Ensure that a library with the given ID exists.
        Raises NotFoundError if the library does not exist.
        
        Args:
            library_id: The ID of the library to check
            
        Raises:
            NotFoundError: If the library does not exist
        """
        try:
            self._repo.get_library(library_id)
        except RepoNotFoundError as e:
            logger.warning(f"Library with ID {library_id} not found")
            raise NotFoundError("Library", str(library_id)) from e

    def add_library(self, library: Library) -> UUID:
        """
        Add a library to the repository.
        Returns the ID of the added library.
        
        Args:
            library: The library to add
            
        Returns:
            The ID of the added library
        """
        logger.info(f"Adding library with ID {library.id}")
        self._metrics("library.add", library_id=str(library.id))
        
        self._repo.add_library(library)
        return library.id

    def get_library(self, library_id: UUID) -> Library:
        """
        Get a library by ID.
        Raises NotFoundError if the library does not exist.
        
        Args:
            library_id: The ID of the library to get
            
        Returns:
            The library with the given ID
            
        Raises:
            NotFoundError: If the library does not exist
        """
        logger.info(f"Getting library with ID {library_id}")
        self._metrics("library.get", library_id=str(library_id))
        
        try:
            return self._repo.get_library(library_id)
        except RepoNotFoundError as e:
            logger.warning(f"Library with ID {library_id} not found")
            raise NotFoundError("Library", str(library_id)) from e

    def update_library(self, library_id: UUID, updated: Library) -> None:
        """
        Update a library.
        Raises NotFoundError if the library does not exist.
        
        Args:
            library_id: The ID of the library to update
            updated: The updated library
            
        Raises:
            NotFoundError: If the library does not exist
        """
        logger.info(f"Updating library with ID {library_id}")
        self._metrics("library.update", library_id=str(library_id))
        
        # Try atomic update first
        if not self._repo.update_library_if_exists(library_id, updated):
            # If atomic update fails, ensure the library exists to get the proper error
            self._ensure_exists(library_id)
            # This should never happen if _ensure_exists doesn't raise
            raise RuntimeError(f"Failed to update library with ID {library_id}")

    def delete_library(self, library_id: UUID) -> None:
        """
        Delete a library by ID.
        Raises NotFoundError if the library does not exist.
        
        Args:
            library_id: The ID of the library to delete
            
        Raises:
            NotFoundError: If the library does not exist
        """
        logger.info(f"Deleting library with ID {library_id}")
        self._metrics("library.delete", library_id=str(library_id))
        
        try:
            self._repo.delete_library(library_id)
        except RepoNotFoundError as e:
            logger.warning(f"Library with ID {library_id} not found")
            raise NotFoundError("Library", str(library_id)) from e
