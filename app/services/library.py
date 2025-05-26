import time
from typing import Any, Protocol
from uuid import UUID

from loguru import logger

from app.domain.models import Library
from app.repos.in_memory import InMemoryRepo
from app.services.exceptions import NotFoundError


class MetricsCallback(Protocol):
    """Protocol for metrics callbacks."""

    def __call__(self, metric_name: str, **kwargs: Any) -> None:
        ...


def noop_metrics_callback(metric_name: str, **kwargs: Any) -> None:
    """Default no-op metrics callback."""
    pass


class LibraryService:
    """
    Service for managing libraries.
    Uses an InMemoryRepo for storage and provides higher-level operations.
    Includes logging and metrics for observability.
    """

    def __init__(
        self,
        repo: InMemoryRepo,
        metrics_callback: MetricsCallback = noop_metrics_callback,
    ) -> None:
        """
        Initialize the library service.

        Args:
            repo: The repository to use for storage
            metrics_callback: Optional callback for metrics
        """
        self._repo = repo
        self._metrics = metrics_callback

    async def _ensure_exists(self, library_id: UUID) -> None:
        """
        Ensure that a library with the given ID exists.
        Raises NotFoundError if the library is not found.

        Args:
            library_id: The ID of the library to check

        Raises:
            NotFoundError: If the library is not found
        """
        try:
            await self._repo.get_library(library_id)
        except NotFoundError as e:
            logger.warning(f"Library with ID {library_id} not found")
            raise NotFoundError(
                message=f"Library with ID {library_id} not found",
                resource_type="Library",
                resource_id=library_id,
            ) from e

    async def add_library(self, library: Library) -> UUID:
        """
        Add a library to the repository.
        Returns the ID of the added lib.

        Args:
            library: The library to add

        Returns:
            The ID of the added lib
        """
        start_time = time.time()
        logger.info(f"Adding library with ID {library.id}")
        try:
            await self._repo.add_library(library)
            self._metrics(
                "library.add",
                duration_ms=(time.time() - start_time) * 1000,
                library_id=str(library.id),
            )
            return library.id
        except Exception as e:
            logger.error(f"Error adding library: {e}")
            self._metrics(
                "library.add_error",
                duration_ms=(time.time() - start_time) * 1000,
                library_id=str(library.id),
            )
            raise  # Re-raise the exception after logging and metrics

    async def get_library(self, library_id: UUID) -> Library:
        """
        Get a library by ID.
        Raises NotFoundError if the library is not found.

        Args:
            library_id: The ID of the library to get

        Returns:
            The library with the given ID

        Raises:
            NotFoundError: If the library is not found
        """
        start_time = time.time()
        logger.info(f"Getting library with ID {library_id}")
        try:
            library = await self._repo.get_library(library_id)
            self._metrics(
                "library.get",
                duration_ms=(time.time() - start_time) * 1000,
                library_id=str(library_id),
            )
            return library
        except NotFoundError as e:
            logger.warning(f"Library with ID {library_id} not found")
            self._metrics(
                "library.get_not_found",
                duration_ms=(time.time() - start_time) * 1000,
                library_id=str(library_id),
            )
            raise NotFoundError(
                message=f"Library with ID {library_id} not found",
                resource_type="Library",
                resource_id=library_id,
            ) from e
        except Exception as e:
            logger.error(f"Error getting library {library_id}: {e}")
            self._metrics(
                "library.get_error",
                duration_ms=(time.time() - start_time) * 1000,
                library_id=str(library_id),
            )
            raise  # Re-raise any other exception

    async def update_library(self, library_id: UUID, updated: Library) -> None:
        """
        Update a library.
        Raises NotFoundError if the library does not exist.

        Args:
            library_id: The ID of the library to update
            updated: The updated library

        Raises:
            NotFoundError: If the library does not exist
        """
        start_time = time.time()
        logger.info(f"Updating library with ID {library_id}")
        try:
            if not await self._repo.update_library_if_exists(library_id, updated):
                await self._ensure_exists(library_id)
                raise RuntimeError(f"Update failed for lib ID {library_id}")
            duration_ms = (time.time() - start_time) * 1000
            self._metrics(
                "library.update",
                duration_ms=duration_ms,
                library_id=str(library_id),
            )
        except NotFoundError as e:
            logger.warning(f"Lib not found for update: {library_id}")
            duration_ms = (time.time() - start_time) * 1000
            self._metrics(
                "library.update_not_found",
                duration_ms=duration_ms,
                library_id=str(library_id),
            )
            raise NotFoundError(
                message=f"Library with ID {library_id} not found",
                resource_type="Library",
                resource_id=library_id,
            ) from e
        except Exception as e:
            logger.error(f"Error updating library {library_id}: {e}")
            duration_ms = (time.time() - start_time) * 1000
            self._metrics(
                "library.update_error",
                duration_ms=duration_ms,
                library_id=str(library_id),
            )
            raise

    async def delete_library(self, library_id: UUID) -> None:
        """
        Delete a library by ID.
        Raises NotFoundError if the library does not exist.

        Args:
            library_id: The ID of the library to delete

        Raises:
            NotFoundError: If the library does not exist
        """
        start_time = time.time()
        logger.info(f"Deleting library with ID {library_id}")
        try:
            await self._repo.get_library(library_id)
            await self._repo.delete_library(library_id)
            duration_ms = (time.time() - start_time) * 1000
            self._metrics(
                "library.delete",
                duration_ms=duration_ms,
                library_id=str(library_id),
            )
        except NotFoundError as e:
            logger.warning(f"Lib not found for deletion: {library_id}")
            duration_ms = (time.time() - start_time) * 1000
            self._metrics(
                "library.delete_not_found",
                duration_ms=duration_ms,
                library_id=str(library_id),
            )
            raise NotFoundError(
                message=f"Library with ID {library_id} not found",
                resource_type="Library",
                resource_id=library_id,
            ) from e
        except Exception as e:
            logger.error(f"Error deleting library {library_id}: {e}")
            duration_ms = (time.time() - start_time) * 1000
            self._metrics(
                "library.delete_error",
                duration_ms=duration_ms,
                library_id=str(library_id),
            )
            raise
