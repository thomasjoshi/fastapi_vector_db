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

# Global indices dictionary to maintain state across requests
_indices: Dict[UUID, Any] = {}
_indices_lock = threading.RLock()  # Global lock for thread safety

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
        # Use the global _indices dictionary and lock for persistence between requests
        # self._indices: Dict[UUID, Any] = {}  # No longer needed as instance variable
        # self._lock = threading.RLock()  # No longer needed as instance variable
        
        # Create a no-op observer function that doesn't call any metric system
        # This completely bypasses the metrics functionality which isn't required
        # for our core vector search functionality
        def no_op_observer(op: str, time_ms: float) -> None:
            # Do nothing - we're not using metrics in this implementation
            pass
            
        self._observer_adapter = no_op_observer

    async def index_library(self, library_id: UUID) -> int:
        """
        Index all chunks in a library.

        Args:
            library_id: ID of the library to index

        Returns:
            Number of chunks indexed

        Raises:
            NotFoundError: If the library does not exist
            ValidationError: If chunks have inconsistent dimensions or invalid embeddings
        """
        from loguru import logger
        start_time = time.time()

        try:
            with _indices_lock:
                # Verify library exists
                library = await self._repo.get_library(library_id)
                
                logger.info(f"Starting indexing for library {library_id}")

                # Find all documents in the library
                documents = library.documents
                logger.info(f"Found {len(documents)} documents in library {library_id}")
                
                # Extract all chunks from documents
                chunks = []
                for document in documents:
                    # Extract chunks from the document
                    chunks.extend(document.chunks)

                logger.info(f"Found {len(chunks)} chunks to index in library {library_id}")

                if not chunks:
                    # Create empty index if no chunks
                    logger.warning(f"No chunks found in library {library_id}, creating empty index")
                    _indices[library_id] = self._index_class(dim=0, observer=self._observer_adapter)
                    return 0

                # Verify all chunks have valid embeddings
                invalid_chunks = [chunk for chunk in chunks if not chunk.embedding]
                if invalid_chunks:
                    chunk_ids = [str(chunk.id) for chunk in invalid_chunks[:5]]
                    logger.error(f"Found {len(invalid_chunks)} chunks with empty embeddings: {', '.join(chunk_ids)}")
                    raise ValidationError(
                        f"Found {len(invalid_chunks)} chunks with empty embeddings. First few: {', '.join(chunk_ids)}"
                    )

                # Get the dimension of embeddings
                dims = set(len(chunk.embedding) for chunk in chunks)
                
                if len(dims) > 1:
                    logger.error(f"Inconsistent embedding dimensions in library: {dims}")
                    raise ValidationError(f"Inconsistent embedding dimensions: {dims}")
                
                dim = dims.pop()
                logger.info(f"Creating index with dimension {dim}")
                
                # Build embeddings list
                embeddings = []
                ids = []
                for chunk in chunks:
                    embeddings.append(chunk.embedding)
                    ids.append(chunk.id)
                
                # Create index with proper dimension
                try:
                    logger.info(f"Creating index of type {self._index_class.__name__} with dimension {dim}")
                    index = self._index_class(dim=dim, observer=self._observer_adapter)
                    
                    # Build the index
                    logger.info(f"Building index with {len(embeddings)} embeddings")
                    index.build(embeddings=embeddings, ids=ids)
                    _indices[library_id] = index
                    logger.info(f"Successfully built index with {len(embeddings)} embeddings")
                except Exception as e:
                    logger.error(f"Failed to create/build index: {str(e)}")
                    logger.exception("Exception details:")
                    raise ValidationError(f"Failed to create/build index: {str(e)}")

                end_time = time.time()
                logger.info(f"Indexed {len(chunks)} chunks in {end_time - start_time:.2f}s")

                return len(chunks)
        except Exception as e:
            logger.error(f"Error during library indexing: {str(e)}")
            raise

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

        with _indices_lock:
            # Verify library exists
            library = await self._repo.get_library(library_id)

            # Check if library is indexed
            if library_id not in _indices:
                raise ValidationError(f"Library {library_id} is not indexed")

            index = _indices[library_id]

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
        with _indices_lock:
            # Remove existing index if it exists
            if library_id in _indices:
                del _indices[library_id]

            # Index the library
            return await self.index_library(library_id)
