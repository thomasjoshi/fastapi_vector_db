"""
Service for managing documents.
This is a stub implementation that will be expanded in future increments.
"""
import time
from typing import List
from uuid import UUID

from app.domain.models import Document
from app.repos.in_memory import InMemoryRepo
from app.services.exceptions import NotFoundError
from app.services.library import MetricsCallback, noop_metrics_callback


class DocumentService:
    """
    Service for managing documents within libraries.
    Uses an InMemoryRepo for storage and provides higher-level operations.

    TODO: Implement in future increment with full CRUD operations for documents.
    """

    def __init__(
        self,
        repo: InMemoryRepo,
        metrics_callback: MetricsCallback = noop_metrics_callback,
    ) -> None:
        """
        Initialize the document service.

        Args:
            repo: The repository to use for storage
            metrics_callback: Optional callback for metrics
        """
        self._repo = repo
        self._metrics = metrics_callback

    async def list_documents(self, library_id: UUID) -> List[Document]:
        """
        List all documents in a library.

        Args:
            library_id: ID of the library to list docs from

        Returns:
            List of documents in the library

        Raises:
            NotFoundError: If the library is not found
        """
        start_time = time.time()
        try:
            # Verify library exists by trying to get it
            library = await self._repo.get_library(library_id)

            # Get documents from the library
            documents = library.documents

            duration_ms = (time.time() - start_time) * 1000
            self._metrics("list_documents", duration_ms=duration_ms)
            return documents
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._metrics("list_documents_error", duration_ms=duration_ms)
            raise NotFoundError(
                f"Library with ID {library_id} not found", "Library", library_id
            ) from e

    async def add_document(self, library_id: UUID, document: Document) -> Document:
        """
        Add a document to a library.

        Args:
            library_id: ID of the library to add doc to
            document: Document to add

        Returns:
            The added document with its ID

        Raises:
            NotFoundError: If the library is not found
        """
        start_time = time.time()
        try:
            # Add document to the library
            await self._repo.add_document(library_id, document)

            duration_ms = (time.time() - start_time) * 1000
            self._metrics("add_document", duration_ms=duration_ms)
            return document
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._metrics("add_document_error", duration_ms=duration_ms)
            raise NotFoundError(
                f"Library with ID {library_id} not found", "Library", library_id
            ) from e

    async def get_document(self, library_id: UUID, document_id: UUID) -> Document:
        """
        Get a document by ID from a library.

        Args:
            library_id: ID of the library containing the document
            document_id: ID of the document to get

        Returns:
            The requested document

        Raises:
            NotFoundError: If the library or document does not exist
        """
        start_time = time.time()
        try:
            # Get document from the library
            document = await self._repo.get_document(library_id, document_id)

            self._metrics("get_document", duration_ms=(time.time() - start_time) * 1000)
            return document
        except Exception as e:
            self._metrics(
                "get_document_error", duration_ms=(time.time() - start_time) * 1000
            )
            raise NotFoundError(
                f"Document with ID {document_id} not found in library {library_id}",
                "Document",
                document_id,
            ) from e

    async def update_document(
        self, library_id: UUID, document_id: UUID, updated: Document
    ) -> Document:
        """
        Update a document in a library.

        Args:
            library_id: ID of the library containing the document
            document_id: ID of the document to update
            updated: Updated document data

        Returns:
            The updated document

        Raises:
            NotFoundError: If the library or document does not exist
        """
        start_time = time.time()
        try:
            # Update document in the repository
            await self._repo.update_document(library_id, document_id, updated)

            # Get the updated document
            document = await self._repo.get_document(library_id, document_id)

            self._metrics(
                "update_document", duration_ms=(time.time() - start_time) * 1000
            )
            return document
        except Exception as e:
            self._metrics(
                "update_document_error", duration_ms=(time.time() - start_time) * 1000
            )
            raise NotFoundError(
                f"Document with ID {document_id} not found in library {library_id}",
                "Document",
                document_id,
            ) from e

    async def delete_document(self, library_id: UUID, document_id: UUID) -> None:
        """
        Delete a document from a library.

        Args:
            library_id: ID of the library containing the document
            document_id: ID of the document to delete

        Raises:
            NotFoundError: If the library or document does not exist
        """
        start_time = time.time()
        try:
            # Delete document from the repository
            await self._repo.delete_document(library_id, document_id)

            self._metrics(
                "delete_document", duration_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            self._metrics(
                "delete_document_error", duration_ms=(time.time() - start_time) * 1000
            )
            raise NotFoundError(
                f"Document with ID {document_id} not found in library {library_id}",
                "Document",
                document_id,
            ) from e
