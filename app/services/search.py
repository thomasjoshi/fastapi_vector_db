"""
Search service for vector similarity search.
"""

import threading
from typing import Dict, List, Type, TypeVar
from uuid import UUID

from app.domain.models import Chunk
from app.indexing import VectorIndex
from app.repos.in_memory import InMemoryRepo
from app.services.errors import NotFoundError

T = TypeVar("T")


class SearchService:
    """
    Service for vector similarity search across libraries.

    This service maintains vector indices for libraries and provides
    methods to index libraries and query for similar chunks.
    """

    def __init__(
        self, repo: InMemoryRepo, index_factory: Type[VectorIndex[UUID]]
    ) -> None:
        """
        Initialize the search service.

        Args:
            repo: Repository for accessing libraries and chunks
            index_factory: Factory function to create vector indices
        """
        self._repo = repo
        self._index_factory = index_factory
        self._indices: Dict[UUID, VectorIndex[UUID]] = {}
        self._lock = threading.RLock()  # Reentrant lock for thread safety

    def index_library(self, library_id: UUID) -> int:
        """
        Index all chunks in a library.

        Args:
            library_id: ID of the library to index

        Returns:
            Number of chunks indexed

        Raises:
            NotFoundError: If the library does not exist
        """
        with self._lock:
            # Get the library from the repository
            library = self._repo.get_library(library_id)

            # Create a new index for this library
            index = self._index_factory()

            # Count the number of chunks indexed
            chunks_indexed = 0

            # Add all chunks to the index
            for document in library.documents:
                for chunk in document.chunks:
                    if chunk.embedding:  # Skip chunks without embeddings
                        index.add(chunk.id, chunk.embedding)
                        chunks_indexed += 1

            # Store the index
            self._indices[library_id] = index

            return chunks_indexed

    def query(
        self, library_id: UUID, embedding: List[float], k: int = 5
    ) -> List[Chunk]:
        """
        Query for similar chunks in a library.

        Args:
            library_id: ID of the library to query
            embedding: Query vector
            k: Number of results to return

        Returns:
            List of chunks, sorted by similarity in descending order

        Raises:
            NotFoundError: If the library does not exist or is not indexed
        """
        with self._lock:
            # Check if the library exists
            library = self._repo.get_library(library_id)

            # Check if the library is indexed
            if library_id not in self._indices:
                raise NotFoundError(f"Library {library_id} is not indexed")

            # Query the index
            index = self._indices[library_id]
            results = index.query(embedding, k)

            # Retrieve the chunks
            chunks = []
            for chunk_id, _ in results:
                # Find the document containing this chunk
                for document in library.documents:
                    for chunk in document.chunks:
                        if chunk.id == chunk_id:
                            chunks.append(chunk)
                            break

            return chunks

    def reindex_library(self, library_id: UUID) -> int:
        """
        Reindex a library, replacing any existing index.

        Args:
            library_id: ID of the library to reindex

        Returns:
            Number of chunks indexed

        Raises:
            NotFoundError: If the library does not exist
        """
        with self._lock:
            # Remove existing index if it exists
            if library_id in self._indices:
                del self._indices[library_id]

            # Index the library
            return self.index_library(library_id)
