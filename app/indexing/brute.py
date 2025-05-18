"""
Brute force cosine similarity search implementation.
"""

import io
import pickle
import threading
import time
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np
from numpy.linalg import norm

T = TypeVar("T")


class DuplicateVectorError(Exception):
    """Raised when attempting to add a vector with an ID that already exists."""

    def __init__(self, id: Any) -> None:
        self.id = id
        super().__init__(f"Vector with ID {id} already exists in the index")


class BruteForceCosine(Generic[T]):
    """
    Brute force cosine similarity search.

    This is the simplest implementation that computes cosine similarity
    between the query vector and all vectors in the index.

    Time complexity:
    - Build: O(n) where n is the number of vectors
    - Query: O(n*d) where n is the number of vectors and d is the dimension

    Space complexity: O(n*d) where n is the number of vectors and d is the dimension

    Thread-safe: All operations are protected by a reentrant lock.
    """

    def __init__(
        self, dim: int, observer: Optional[Callable[[str, float], None]] = None
    ) -> None:
        """
        Initialize an empty index.

        Args:
            dim: Dimensionality of vectors to be indexed
            observer: Optional callback for performance metrics
        """
        self._dim = dim
        self._matrix = np.zeros((0, dim), dtype=np.float32)
        self._ids: List[T] = []
        self._lock = threading.RLock()
        self._observer = observer or (lambda op, time_ms: None)

    def add(self, id: T, embedding: Sequence[float], overwrite: bool = False) -> None:
        """
        Add a vector to the index.

        Args:
            id: Unique identifier for the vector
            embedding: Vector to add to the index
            overwrite: If True, overwrite existing vector with same ID;
                      if False, raise DuplicateVectorError

        Raises:
            ValueError: If embedding dimension doesn't match index dimension
            DuplicateVectorError: If ID already exists and overwrite=False
        """
        start_time = time.time()

        # Validate embedding dimension
        if len(embedding) != self._dim:
            msg = f"Expected dimension {self._dim}, got {len(embedding)}"
            raise ValueError(msg)

        with self._lock:
            # Check for duplicate ID
            try:
                idx = self._ids.index(id)
                if not overwrite:
                    raise DuplicateVectorError(id)
                # Overwrite existing vector
                vector = np.array(embedding, dtype=np.float32)
                vector_norm = norm(vector)
                if vector_norm > 0:
                    normalized = vector / vector_norm
                else:
                    normalized = vector
                self._matrix[idx] = normalized
            except ValueError:
                # ID not found, add new vector
                vector = np.array(embedding, dtype=np.float32)
                vector_norm = norm(vector)
                if vector_norm > 0:
                    normalized = vector / vector_norm
                else:
                    normalized = vector

                # Append to matrix and IDs list
                self._matrix = np.vstack([self._matrix, normalized[np.newaxis, :]])
                self._ids.append(id)

        duration_ms = (time.time() - start_time) * 1000
        self._observer("add", duration_ms)

    def build(
        self, embeddings: Sequence[Sequence[float]], ids: Optional[Sequence[T]] = None
    ) -> None:
        """
        Build the index from a list of embeddings.

        Args:
            embeddings: List of vectors to add to the index
            ids: Optional list of IDs corresponding to the embeddings.
                 If not provided, indices will be used as IDs.

        Raises:
            ValueError: If embeddings have inconsistent dimensions or don't match
                the index dimension
        """
        start_time = time.time()

        if not embeddings:
            return

        # Validate all embeddings have the same dimension
        for i, emb in enumerate(embeddings):
            if len(emb) != self._dim:
                msg = f"Embedding {i} has dimension {len(emb)}, expected {self._dim}"
                raise ValueError(msg)

        if ids is None:
            ids = list(range(len(embeddings)))  # type: ignore

        if len(embeddings) != len(ids):
            raise ValueError("Number of embeddings must match number of IDs")

        with self._lock:
            # Convert to numpy array in one operation
            vectors = np.array(embeddings, dtype=np.float32)

            # Normalize all vectors in one operation
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1.0
            normalized_vectors = vectors / norms

            # Check for duplicate IDs
            id_set = set(self._ids)
            new_ids: List[T] = []
            new_vectors = []

            for i, id_val in enumerate(ids):
                if id_val not in id_set:
                    new_ids.append(id_val)
                    new_vectors.append(normalized_vectors[i])

            if new_vectors:
                new_matrix = np.array(new_vectors, dtype=np.float32)
                if self._matrix.shape[0] == 0:
                    self._matrix = new_matrix
                else:
                    self._matrix = np.vstack([self._matrix, new_matrix])
                self._ids.extend(new_ids)

        duration_ms = (time.time() - start_time) * 1000
        self._observer("build", duration_ms)

    def query(self, embedding: Sequence[float], k: int = 5) -> List[Tuple[T, float]]:
        """
        Query the index for the k nearest neighbors.

        Args:
            embedding: Query vector
            k: Number of nearest neighbors to return

        Returns:
            List of (id, similarity) tuples, sorted by similarity in descending order

        Raises:
            ValueError: If embedding dimension doesn't match index dimension
        """
        start_time = time.time()

        # Validate embedding dimension
        if len(embedding) != self._dim:
            msg = f"Expected dimension {self._dim}, got {len(embedding)}"
            raise ValueError(msg)

        with self._lock:
            if self._matrix.shape[0] == 0:
                return []

            # Normalize the query vector
            query_vector = np.array(embedding, dtype=np.float32)
            query_norm = norm(query_vector)
            if query_norm > 0:
                query_vector = query_vector / query_norm

            # Vectorized similarity computation
            similarities = self._matrix @ query_vector

            # Get top k indices
            if k >= len(self._ids):
                # If k is larger than the number of vectors, return all
                top_indices = np.argsort(similarities)[::-1]
            else:
                # Otherwise, get top k
                top_indices = np.argsort(similarities)[-k:][::-1]

            # Create result list
            results = [(self._ids[i], float(similarities[i])) for i in top_indices]

        duration_ms = (time.time() - start_time) * 1000
        self._observer("query", duration_ms)

        return results

    def size(self) -> int:
        """Return the number of vectors in the index."""
        with self._lock:
            return len(self._ids)

    def to_bytes(self) -> bytes:
        """
        Serialize the index to bytes.

        Returns:
            Serialized index as bytes
        """
        with self._lock:
            # Create a BytesIO object to write the data
            buffer = io.BytesIO()

            # Save the matrix
            np.save(buffer, self._matrix)

            # Save the dimension
            np.save(buffer, np.array([self._dim], dtype=np.int32))

            # Save the IDs
            pickle.dump(self._ids, buffer)

            # Get the bytes
            buffer.seek(0)
            return buffer.getvalue()

    @classmethod
    def from_bytes(
        cls, data: bytes, observer: Optional[Callable[[str, float], None]] = None
    ) -> "BruteForceCosine[T]":
        """
        Deserialize an index from bytes.

        Args:
            data: Serialized index as bytes
            observer: Optional callback for performance metrics

        Returns:
            Deserialized BruteForceCosine index
        """
        buffer = io.BytesIO(data)

        # Load the matrix
        matrix = np.load(buffer)

        # Load the dimension
        dim = int(np.load(buffer)[0])

        # Load the IDs
        ids = pickle.load(buffer)

        # Create a new index
        index = cls(dim, observer)

        # Set the data
        index._matrix = matrix
        index._ids = ids

        return index
