"""
Vector indexing implementations for efficient similarity search.
"""

from typing import Any, List, Protocol, Tuple, TypeVar

T = TypeVar("T")


class VectorIndex(Protocol[T]):
    """Protocol defining the interface for vector indexes."""

    def __init__(self, dim: int, *args: Any, **kwargs: Any) -> None:
        ...

    def add(self, id: T, embedding: List[float]) -> None:
        """Add a vector to the index."""
        ...

    def build(self, embeddings: List[List[float]], ids: List[T] | None = None) -> None:
        """Build the index from a list of embeddings."""
        ...

    def query(self, embedding: List[float], k: int = 5) -> List[Tuple[T, float]]:
        """Query the index for the k nearest neighbors."""
        ...

    def size(self) -> int:
        """Return the number of vectors in the index."""
        ...
