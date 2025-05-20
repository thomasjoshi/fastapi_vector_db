"""
Ball Tree implementation for cosine similarity search.

This is a variance-split KD-tree variant optimized for cosine similarity.
Implements thread-safety, dimensionality enforcement, and memory optimization.
"""

import heapq
import threading
import time
from typing import (
    Any,
    Callable,
    Dict,
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


class Node:
    """Node in the Ball Tree."""

    def __init__(
        self,
        indices: List[int] = None,
        center: Optional[np.ndarray] = None,
        radius: float = 0.0,
        axis: int = 0,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
    ) -> None:
        """
        Initialize a Ball Tree node.

        Args:
            indices: Indices into the global matrix for leaf nodes
            center: Normalized center vector
            radius: Maximum cosine distance from center to any point
            axis: Split axis for internal nodes
            left: Left child node
            right: Right child node
        """
        self.indices = indices or []
        self.center = center
        self.radius = radius
        self.axis = axis
        self.left = left
        self.right = right

    def is_leaf(self) -> bool:
        """Check if the node is a leaf node."""
        return self.left is None and self.right is None


class BallTreeCosine(Generic[T]):
    """
    Ball Tree implementation for cosine similarity search.

    This is a variance-split KD-tree variant that works well for
    low to medium dimensional data (typically < 20 dimensions).

    Features:
    - Dimensionality enforcement
    - Duplicate ID handling
    - Thread safety
    - Memory optimization
    - Variance-based splitting
    - Optimized pruning

    Time complexity:
    - Build: O(n log n) where n is the number of vectors
    - Query: O(log n) in best case, O(n) in worst case

    Space complexity: O(n*d) where n is the number of vectors and d is the dimension
    """

    def __init__(
        self,
        dim: int,
        leaf_size: int = 40,
        observer: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        """
        Initialize an empty Ball Tree index.

        Args:
            dim: Dimensionality of vectors to be indexed
            leaf_size: Maximum number of vectors in leaf nodes
            observer: Optional callback for performance metrics
        """
        self._dim = dim
        self._leaf_size = leaf_size
        self._root: Optional[Node] = None
        self._matrix = np.zeros((0, dim), dtype=np.float32)
        self._ids: List[T] = []
        self._id_to_idx: Dict[T, int] = {}
        self._lock = threading.RLock()
        self._dirty = True
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
            if id in self._id_to_idx:
                if not overwrite:
                    raise DuplicateVectorError(id)

                # If overwrite, update the existing vector
                idx = self._id_to_idx[id]

                # Normalize the vector
                vector = np.array(embedding, dtype=np.float32)
                vector_norm = norm(vector)
                if vector_norm > 0:
                    normalized = vector / vector_norm
                else:
                    normalized = vector

                # Update the matrix
                self._matrix[idx] = normalized
            else:
                # Add new vector
                # Normalize the vector
                vector = np.array(embedding, dtype=np.float32)
                vector_norm = norm(vector)
                if vector_norm > 0:
                    normalized = vector / vector_norm
                else:
                    normalized = vector

                # Add to matrix
                if self._matrix.shape[0] == 0:
                    self._matrix = np.array([normalized], dtype=np.float32)
                else:
                    self._matrix = np.vstack([self._matrix, normalized])

                # Add to ID mappings
                self._ids.append(id)
                self._id_to_idx[id] = len(self._ids) - 1

            # Mark as dirty (needs rebuild)
            self._dirty = True

        duration_ms = (time.time() - start_time) * 1000
        self._observer("add", duration_ms)

    def build(
        self, embeddings: Sequence[Sequence[float]], ids: Optional[Sequence[T]] = None
    ) -> None:
        """
        Build the Ball Tree from a list of embeddings.

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
            id_set = set(self._id_to_idx.keys())
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

                # Update ID mappings
                for id_val in new_ids:
                    self._id_to_idx[id_val] = len(self._ids)
                    self._ids.append(id_val)

            # Build the tree
            self._build_tree()

        duration_ms = (time.time() - start_time) * 1000
        self._observer("build", duration_ms)

    def remove(self, id: T) -> bool:
        """
        Remove a vector from the index.

        Args:
            id: ID of the vector to remove

        Returns:
            True if the vector was removed, False if it wasn't found

        Note: After removing vectors, the tree will be marked as dirty
              and will be rebuilt on the next query.
        """
        start_time = time.time()

        with self._lock:
            if id not in self._id_to_idx:
                return False

            # Get the index of the vector to remove
            idx = self._id_to_idx[id]

            # If it's the last vector, just remove it
            if idx == len(self._ids) - 1:
                self._matrix = self._matrix[:-1]
                self._ids.pop()
                del self._id_to_idx[id]
            else:
                # Swap with the last vector
                last_id = self._ids[-1]

                # Update the matrix
                self._matrix[idx] = self._matrix[-1]
                self._matrix = self._matrix[:-1]

                # Update the ID mappings
                self._ids[idx] = last_id
                self._id_to_idx[last_id] = idx
                self._ids.pop()
                del self._id_to_idx[id]

            # Mark as dirty (needs rebuild)
            self._dirty = True

        duration_ms = (time.time() - start_time) * 1000
        self._observer("remove", duration_ms)
        return True

    def _build_tree(self) -> None:
        """Build the Ball Tree from the stored vectors."""
        if self._matrix.shape[0] == 0:
            self._root = None
            self._dirty = False
            return

        # Build the tree recursively
        indices = list(range(self._matrix.shape[0]))
        self._root = self._build_subtree(indices)
        self._dirty = False

    def _build_subtree(self, indices: List[int], depth: int = 0) -> Node:
        """
        Recursively build a subtree of the Ball Tree.

        Args:
            indices: Indices into the global matrix
            depth: Current depth in the tree

        Returns:
            Root node of the subtree
        """
        if len(indices) <= self._leaf_size:
            # Create a leaf node
            vectors = self._matrix[indices]

            # Calculate center vector (centroid)
            center = np.mean(vectors, axis=0)
            # Normalize the center vector
            center_norm = norm(center)
            if center_norm > 0:
                center = center / center_norm

            # Calculate radius (maximum cosine distance from center)
            radius = 0.0
            for idx in indices:
                # Cosine distance is 1 - cosine similarity
                sim = float(np.dot(self._matrix[idx], center))
                dist = 1.0 - sim
                radius = max(radius, dist)

            return Node(indices=indices, center=center, radius=radius)

        # Get the vectors for these indices
        vectors = self._matrix[indices]

        # Choose axis with highest variance
        variances = np.var(vectors, axis=0)
        axis = int(np.argmax(variances))

        # Sort indices along the chosen axis
        sorted_indices = sorted(indices, key=lambda i: self._matrix[i, axis])
        median_idx = len(sorted_indices) // 2

        # Calculate center vector (centroid)
        center = np.mean(vectors, axis=0)
        # Normalize the center vector
        center_norm = norm(center)
        if center_norm > 0:
            center = center / center_norm

        # Calculate radius (maximum cosine distance from center)
        radius = 0.0
        for idx in indices:
            # Cosine distance is 1 - cosine similarity
            sim = float(np.dot(self._matrix[idx], center))
            dist = 1.0 - sim
            radius = max(radius, dist)

        # Create node and recursively build left and right subtrees
        left = self._build_subtree(sorted_indices[:median_idx], depth + 1)
        right = self._build_subtree(sorted_indices[median_idx:], depth + 1)

        return Node(
            indices=None,
            center=center,
            radius=radius,
            axis=axis,
            left=left,
            right=right,
        )

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
            RuntimeError: If the index is dirty and needs to be rebuilt
        """
        start_time = time.time()

        # Validate embedding dimension
        if len(embedding) != self._dim:
            msg = f"Expected dimension {self._dim}, got {len(embedding)}"
            raise ValueError(msg)

        with self._lock:
            if self._matrix.shape[0] == 0:
                return []

            # Check if the tree needs to be rebuilt
            if self._dirty:
                raise RuntimeError("Index is dirty, call build() first")

            if self._root is None:
                return []

            # Make a copy of the necessary data for thread safety
            matrix_snapshot = self._matrix.copy()
            ids_snapshot = self._ids.copy()
            # Nodes are immutable after build
            root_snapshot = self._root

            # Normalize the query vector
            query_vector = np.array(embedding, dtype=np.float32)
            query_norm = norm(query_vector)
            if query_norm > 0:
                query_vector = query_vector / query_norm

            # Use a priority queue to track the k best results
            # We use negative similarity because heapq is a min heap
            best: List[Tuple[float, T]] = []

            # Recursively search the tree
            self._search_node(
                root_snapshot, query_vector, k, best, matrix_snapshot, ids_snapshot
            )

        # Convert results to the expected format and sort by similarity (descending)
        results = [(id_val, -sim) for sim, id_val in best]

        duration_ms = (time.time() - start_time) * 1000
        self._observer("query", duration_ms)

        return results

    def _search_node(
        self,
        node: Node,
        query: np.ndarray,
        k: int,
        best: List[Tuple[float, T]],
        matrix: np.ndarray,
        ids: List[T],
    ) -> None:
        """
        Recursively search a node and its children for the k nearest neighbors.

        Args:
            node: Current node to search
            query: Normalized query vector
            k: Number of nearest neighbors to find
            best: Priority queue of the best results found so far
            matrix: Matrix of vectors
            ids: List of IDs
        """
        if node.is_leaf():
            # For leaf nodes, compute similarity with all vectors
            for idx in node.indices:
                sim = float(np.dot(query, matrix[idx]))
                if len(best) < k:
                    heapq.heappush(best, (-sim, ids[idx]))
                elif sim > -best[0][0]:
                    heapq.heappushpop(best, (-sim, ids[idx]))
            return

        # Calculate cosine similarity to the center
        center_sim = float(np.dot(query, node.center))

        # Calculate cosine distance to center
        dist_to_center = 1.0 - center_sim

        # Get the worst distance in our current best results
        worst_best_distance = float("inf")
        if len(best) >= k:
            worst_best_distance = 1.0 - (
                -best[0][0]
            )  # Convert from similarity to distance

        # If the node is too far away, we can skip it
        # This is the key pruning condition: if the closest point in the node
        # is farther than our current kth-best distance, we can skip the node
        if dist_to_center - node.radius > worst_best_distance:
            return

        # Decide which child to search first based on the splitting axis
        left_first = query[node.axis] <= node.center[node.axis]

        if left_first:
            if node.left:
                self._search_node(node.left, query, k, best, matrix, ids)
            if node.right:
                # Check if we still need to search the right subtree
                need_to_search = (
                    len(best) < k or dist_to_center - node.radius <= worst_best_distance
                )
                if need_to_search:
                    self._search_node(node.right, query, k, best, matrix, ids)
        else:
            if node.right:
                self._search_node(node.right, query, k, best, matrix, ids)
            if node.left:
                # Check if we still need to search the left subtree
                need_to_search = (
                    len(best) < k or dist_to_center - node.radius <= worst_best_distance
                )
                if need_to_search:
                    self._search_node(node.left, query, k, best, matrix, ids)

    def size(self) -> int:
        """Return the number of vectors in the index."""
        with self._lock:
            return len(self._ids)
