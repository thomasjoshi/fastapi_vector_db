"""
Search service for vector similarity search.
"""

import threading
import time
from typing import Any, Dict, Generic, List, Tuple, Type, TypeVar
from uuid import UUID

from app.domain.models import Chunk
from app.indexing.linear_search import LinearSearchCosine
from app.repos.in_memory import InMemoryRepo
from app.services.exceptions import ValidationError

T = TypeVar("T")


class SearchService(Generic[T]):
    """
    Service for vector similarity search across libraries.

    This service maintains vector indices for libraries and provides
    methods to index libraries and query for similar chunks.
    """

    def __init__(
        self, repo: InMemoryRepo, index_class: Type = LinearSearchCosine
    ) -> None:
        """
        Initialize the search service.

        Args:
            repo: Repository for accessing libraries and chunks
            index_class: Vector index implementation class to use (default: LinearSearchCosine)
        """
        self._repo = repo
        self._index_class = index_class
        self._indices: Dict[UUID, Any] = {}
        self._lock = threading.RLock()  # Reentrant lock for thread safety

    async def index_library(self, library_id: UUID) -> int:
        """
        Index all chunks in a library.

        Args:
            library_id: ID of the library to index

        Returns:
            Number of chunks indexed

        Raises:
            NotFoundError: If the library does not exist
            ValidationError: If chunks have inconsistent dimensions
        """
        start_time = time.time()

        with self._lock:
            # Verify library exists
            library = await self._repo.get_library(library_id)

            # Get all chunks in the library
            chunks: List[Chunk] = []
            # Since we don't have list_documents, we'll extract them from the library
            documents = library.documents

            for document in documents:
                # Since we don't have list_chunks, we'll extract them from the document
                chunks.extend(document.chunks)

            if not chunks:
                # Create empty index if no chunks
                self._indices[library_id] = self._index_class[UUID](dim=0)
                return 0

            # Determine dimension from first chunk
            dim = len(chunks[0].embedding)

            # Create index
            index = self._index_class[UUID](dim=dim)

            # Add all chunks to index
            embeddings = []
            ids = []

            for chunk in chunks:
                if len(chunk.embedding) != dim:
                    raise ValidationError(
                        f"Chunk {chunk.id} has embedding dimension {len(chunk.embedding)}, expected {dim}"
                    )
                embeddings.append(chunk.embedding)
                ids.append(chunk.id)

            # Build the index
            index.build(embeddings, ids)
            self._indices[library_id] = index

            end_time = time.time()
            print(f"Indexed {len(chunks)} chunks in {end_time - start_time:.2f}s")

            return len(chunks)

    async def query(
        self, library_id: UUID, embedding: List[float], k: int = 5
    ) -> List[Chunk]:
        """
        Search for chunks in a library and return just the chunks (without scores).

        Args:
            library_id: ID of the library to search in
            embedding: Query vector
            k: Number of results to return

        Returns:
            List of chunks sorted by similarity to the query vector

        Raises:
            NotFoundError: If the library does not exist
            ValidationError: If the library is not indexed or query vector has wrong dimension
        """
        # Call search and extract just the chunks
        results = await self.search(library_id, embedding, k)
        return [chunk for chunk, _ in results]

    async def search(
        self, library_id: UUID, embedding: List[float], k: int = 5, 
        metadata_filters: Dict[str, str] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for similar chunks in a library.

        Args:
            library_id: ID of the library to search in
            embedding: Query vector
            k: Number of results to return
            metadata_filters: Optional dictionary of metadata key-value pairs to filter results

        Returns:
            List of (chunk, score) tuples, sorted by score in descending order

        Raises:
            NotFoundError: If the library does not exist
            ValidationError: If the library is not indexed or query vector has wrong dimension
        """
        start_time = time.time()

        with self._lock:
            # Verify library exists
            library = await self._repo.get_library(library_id)

            # Check if library is indexed
            if library_id not in self._indices:
                raise ValidationError(f"Library {library_id} is not indexed")

            index = self._indices[library_id]

            # Check embedding dimension
            if index.size() > 0 and len(embedding) != index._dim:
                raise ValidationError(
                    f"Query embedding dimension {len(embedding)} does not match index dimension {index._dim}"
                )

            # Search index
            results = index.query(embedding, k=k)

            # Fetch chunks
            hits = []
            # Dictionary for quick chunk lookup
            chunk_map = {}
            for doc in library.documents:
                for c in doc.chunks:
                    chunk_map[c.id] = c
            
            for chunk_id, score in results:
                # Retrieve the chunk from the map
                chunk = chunk_map.get(chunk_id)
                
                if chunk:  # Skip if chunk was deleted
                    # Apply metadata filters if provided
                    if metadata_filters:
                        # Check if chunk metadata matches all filter criteria
                        matches = True
                        for key, value in metadata_filters.items():
                            # Skip chunks that don't match any filter criteria
                            if key not in chunk.metadata or chunk.metadata[key] != value:
                                matches = False
                                break
                        
                        # Only add chunk if it matches all filters
                        if matches:
                            hits.append((chunk, score))
                    else:
                        # No filters, add all chunks
                        hits.append((chunk, score))

            end_time = time.time()
            print(f"Search completed in {end_time - start_time:.4f}s")

            return hits

    async def reindex_library(self, library_id: UUID) -> int:
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
            return await self.index_library(library_id)
