"""
FastAPI dependency injection functions.
"""


from app.repos.in_memory import InMemoryRepo
from app.services.library import LibraryService, MetricsCallback, noop_metrics_callback

# Singleton instances for the application lifetime
_repo = InMemoryRepo()
_metrics_callback = noop_metrics_callback


def get_repo() -> InMemoryRepo:
    """
    Dependency that provides the repository instance.

    Returns:
        An instance of InMemoryRepo
    """
    return _repo


def get_metrics_callback() -> MetricsCallback:
    """
    Dependency that provides the metrics callback.

    Returns:
        A metrics callback function
    """
    return _metrics_callback


def get_library_service(repo=None, metrics_callback=None) -> LibraryService:
    """
    Dependency that provides a LibraryService instance.

    Args:
        repo: The repository to use
        metrics_callback: The metrics callback to use

    Returns:
        An instance of LibraryService
    """
    if repo is None:
        repo = get_repo()
    if metrics_callback is None:
        metrics_callback = get_metrics_callback()
    return LibraryService(repo, metrics_callback)
