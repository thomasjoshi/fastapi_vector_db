"""
Service for managing chunks.
This is a stub implementation that will be expanded in future increments.
"""
from typing import List
from uuid import UUID

from loguru import logger

from app.domain.models import Chunk
from app.repos.in_memory import InMemoryRepo
from app.services.exceptions import NotFoundError
from app.services.library import MetricsCallback, noop_metrics_callback


class ChunkService:
    """
    Service for managing chunks within documents.
    Uses an InMemoryRepo for storage and provides higher-level operations.

    TODO: Implement in future increment with full CRUD operations for chunks.
    """

    def __init__(
        self,
        repo: InMemoryRepo,
        metrics_callback: MetricsCallback = noop_metrics_callback,
    ) -> None:
        """
        Initialize the chunk service.

        Args:
            repo: The repository to use for storage
            metrics_callback: Optional callback for metrics collection
        """
        self._repo = repo
        self._metrics = metrics_callback

    async def list_chunks(self, library_id: UUID, document_id: UUID) -> List[Chunk]:
        """
        List all chunks in a document.

        Args:
            library_id: ID of the library containing the document
            document_id: ID of the document to list chunks from

        Returns:
            List of chunks in the document

        Raises:
            NotFoundError: If the library or document does not exist
        """
        try:
            # Verify document exists and belongs to library
            # This will raise NotFoundError if document doesn't exist in the library
            document = await self._repo.get_document(library_id, document_id)

            # Get chunks by accessing the document's chunks attribute
            chunks = document.chunks

            self._metrics("list_chunks")
            return chunks
        except Exception as e:
            self._metrics("list_chunks_error")
            if isinstance(e, NotFoundError):
                raise e
            raise NotFoundError(
                f"Document {document_id} not found in library {library_id}",
                "Document",
                document_id,
            ) from e

    async def add_chunk(
        self, library_id: UUID, document_id: UUID, chunk: Chunk
    ) -> Chunk:
        """
        Add a chunk to a document.

        Args:
            library_id: ID of the library containing the document
            document_id: ID of the document to add the chunk to
            chunk: Chunk to add

        Returns:
            The added chunk with its ID

        Raises:
            NotFoundError: If the library or document does not exist
        """
        try:
            # Create new chunk with document_id; can't modify frozen instance
            new_chunk = Chunk(
                id=chunk.id,
                text=chunk.text,
                embedding=chunk.embedding,
                metadata=chunk.metadata,
                document_id=document_id,
            )

            # Add chunk to repository - pass all required parameters
            await self._repo.add_chunk(library_id, document_id, new_chunk)

            self._metrics("add_chunk")
            return new_chunk
        except Exception as e:
            self._metrics("add_chunk_error")
            if isinstance(e, NotFoundError):
                raise e
            raise NotFoundError(
                f"Doc {document_id} not found in lib {library_id}",
                "Document",
                document_id,
            ) from e

    async def get_chunk(
        self, library_id: UUID, document_id: UUID, chunk_id: UUID
    ) -> Chunk:
        """
        Get a chunk by ID from a document.

        Args:
            library_id: ID of the library containing the document
            document_id: ID of the document containing the chunk
            chunk_id: ID of the chunk to get

        Returns:
            The requested chunk

        Raises:
            NotFoundError: If the library, document, or chunk does not exist
        """
        try:
            await self._repo.get_document(library_id, document_id)

            # Get chunk by passing all required IDs
            chunk = await self._repo.get_chunk(library_id, document_id, chunk_id)
            # The check `if not chunk or chunk.document_id != document_id:`
            # might be redundant if repo.get_chunk already ensures this, 
            # but it's good for defense. With document_id now in Chunk, this check is valid.
            if not chunk: # repo.get_chunk raises NotFoundError, so this might not be strictly needed
                raise NotFoundError(
                    f"Chunk {chunk_id} not found in document {document_id}",
                    "Chunk",
                    chunk_id,
                )
            # Ensure chunk.document_id matches, if repo doesn't guarantee it
            # (InMemoryRepo.get_chunk does seem to guarantee it by finding it within the specific document)

            self._metrics("get_chunk")
            return chunk
        except Exception as e:
            self._metrics("get_chunk_error")
            if isinstance(e, NotFoundError):
                raise e
            raise NotFoundError(
                f"Error retrieving chunk {chunk_id}", "Chunk", chunk_id
            ) from e

    async def update_chunk(
        self, library_id: UUID, document_id: UUID, chunk_id: UUID, updated: Chunk
    ) -> Chunk:
        """
        Update a chunk in a document.

        Args:
            library_id: ID of the library containing the document
            document_id: ID of the document containing the chunk
            chunk_id: ID of the chunk to update
            updated: Updated chunk data

        Returns:
            The updated chunk

        Raises:
            NotFoundError: If the library, document, or chunk does not exist
        """
        # Verify chunk exists in the document in the library
        chunk = await self.get_chunk(library_id, document_id, chunk_id)

        # Create a new chunk with updated values since we can't modify frozen instance
        new_chunk = Chunk(
            id=chunk.id,
            text=updated.text,
            embedding=updated.embedding,
            metadata=updated.metadata,
            document_id=document_id,
        )

        # Update in repository by passing all required IDs and the updated chunk object
        await self._repo.update_chunk(library_id, document_id, chunk_id, new_chunk)

        self._metrics("update_chunk")
        return new_chunk

    async def delete_chunk(
        self, library_id: UUID, document_id: UUID, chunk_id: UUID
    ) -> None:
        """
        Delete a chunk from a document.

        Args:
            library_id: ID of the library containing the document
            document_id: ID of the document containing the chunk
            chunk_id: ID of the chunk to delete

        Raises:
            NotFoundError: If the library, document, or chunk does not exist
        """
        # Verify chunk exists in the document in the library
        await self.get_chunk(library_id, document_id, chunk_id)

        # Delete chunk by passing all required IDs
        await self._repo.delete_chunk(library_id, document_id, chunk_id)

        self._metrics("delete_chunk")

    async def search_chunks(
        self, library_id: UUID, query_embedding: List[float], top_k: int
    ) -> List[Chunk]:
        """Search for similar chunks within a library."""
        # This method is not implemented in ChunkService
        logger.warning("Use SearchService for search functionality.")
        raise NotImplementedError(
            "Search functionality is implemented in SearchService"
        )
