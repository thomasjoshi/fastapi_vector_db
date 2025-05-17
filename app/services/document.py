"""
Service for managing documents.
This is a stub implementation that will be expanded in future increments.
"""
from uuid import UUID

from app.domain.models import Document
from app.repos.in_memory import InMemoryRepo
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
            metrics_callback: Optional callback for metrics collection
        """
        self._repo = repo
        self._metrics = metrics_callback

    # TODO: Implement the following methods in future increments:
    # - add_document(library_id, document)
    # - get_document(library_id, document_id)
    # - update_document(library_id, document_id, updated)
    # - delete_document(library_id, document_id)
    # - list_documents(library_id, filter_criteria=None, pagination=None)

    def add_document(self, library_id: UUID, document: Document) -> UUID:
        """
        Add a document to a library.

        TODO: Implement in future increment.
        """
        pass

    def get_document(self, library_id: UUID, document_id: UUID) -> Document:
        """
        Get a document by ID from a library.

        TODO: Implement in future increment.
        """
        pass

    def update_document(
        self, library_id: UUID, document_id: UUID, updated: Document
    ) -> None:
        """
        Update a document in a library.

        TODO: Implement in future increment.
        """
        pass

    def delete_document(self, library_id: UUID, document_id: UUID) -> None:
        """
        Delete a document from a library.

        TODO: Implement in future increment.
        """
        pass
