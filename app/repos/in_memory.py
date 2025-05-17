from contextlib import contextmanager
from threading import Condition
from time import time
from typing import Callable, Dict, List, Optional, Protocol, TypeVar
from uuid import UUID

from app.domain.models import Chunk, Document, Library


class MetricsCallback(Protocol):
    def __call__(self, lock_type: str, duration_ms: float) -> None: ...


def noop_metrics_callback(lock_type: str, duration_ms: float) -> None:
    """Default no-op metrics callback."""
    pass


class NotFoundError(KeyError):
    """Raised when an entity is not found in the repository."""
    pass


class ReaderWriterLock:
    """
    A reader-writer lock that allows multiple concurrent reads but exclusive writes.
    Uses threading.Condition to coordinate access.
    Provides context managers for read and write locks to ensure proper release.
    Implements fairness for writers to prevent writer starvation.
    """

    def __init__(self, metrics_callback: MetricsCallback = noop_metrics_callback) -> None:
        self._condition = Condition()
        self._active_readers = 0
        self._active_writers = 0
        self._waiting_writers = 0
        self._metrics_callback = metrics_callback

    def acquire_read(self) -> None:
        """Acquire a read lock. Multiple readers can hold the lock simultaneously."""
        start_time = time()
        with self._condition:
            # Wait if there are active writers or waiting writers (fairness for writers)
            while self._active_writers > 0 or self._waiting_writers > 0:
                self._condition.wait()
            self._active_readers += 1
        duration_ms = (time() - start_time) * 1000
        self._metrics_callback("read", duration_ms)

    def release_read(self) -> None:
        """Release a read lock."""
        with self._condition:
            self._active_readers -= 1
            # If this was the last reader, notify waiting writers first
            if self._active_readers == 0:
                self._condition.notify_all()

    def acquire_write(self) -> None:
        """Acquire a write lock. Only one writer can hold the lock at a time."""
        start_time = time()
        with self._condition:
            self._waiting_writers += 1
            # Wait until there are no active readers or writers
            while self._active_readers > 0 or self._active_writers > 0:
                self._condition.wait()
            self._waiting_writers -= 1
            self._active_writers = 1
        duration_ms = (time() - start_time) * 1000
        self._metrics_callback("write", duration_ms)

    def release_write(self) -> None:
        """Release a write lock."""
        with self._condition:
            self._active_writers = 0
            # Notify all waiting threads (readers and writers)
            self._condition.notify_all()
            
    @contextmanager
    def read_lock(self):
        """Context manager for read lock."""
        try:
            self.acquire_read()
            yield
        finally:
            self.release_read()
            
    @contextmanager
    def write_lock(self):
        """Context manager for write lock."""
        try:
            self.acquire_write()
            yield
        finally:
            self.release_write()


T = TypeVar('T', Library, Document, Chunk)


class InMemoryRepo:
    """
    Thread-safe in-memory repository for Library, Document, and Chunk objects.
    Uses a ReaderWriterLock to allow concurrent reads but exclusive writes.
    Provides extended CRUD operations for all entity types with explicit feedback.
    """

    def __init__(self, metrics_callback: MetricsCallback = noop_metrics_callback) -> None:
        self._lock = ReaderWriterLock(metrics_callback)
        self._libraries: Dict[UUID, Library] = {}

    # Library CRUD operations
    
    def add_library(self, library: Library) -> bool:
        """
        Add a library to the repository.
        If a library with the same ID already exists, it will be overwritten.
        Returns True if the library was added, False if it was updated.
        """
        with self._lock.write_lock():
            exists = library.id in self._libraries
            self._libraries[library.id] = library
            return not exists

    def get_library(self, library_id: UUID) -> Library:
        """
        Get a library by ID.
        Raises NotFoundError if the library does not exist.
        """
        with self._lock.read_lock():
            if library_id not in self._libraries:
                raise NotFoundError(f"Library with ID {library_id} not found")
            return self._libraries[library_id]

    def update_library(self, library_id: UUID, updated: Library) -> bool:
        """
        Update a library.
        Raises NotFoundError if the library does not exist.
        Returns True if the library was updated.
        """
        with self._lock.write_lock():
            if library_id not in self._libraries:
                raise NotFoundError(f"Library with ID {library_id} not found")
            self._libraries[library_id] = updated
            return True

    def delete_library(self, library_id: UUID) -> bool:
        """
        Delete a library by ID.
        Raises NotFoundError if the library does not exist.
        Returns True if the library was deleted.
        """
        with self._lock.write_lock():
            if library_id not in self._libraries:
                raise NotFoundError(f"Library with ID {library_id} not found")
            del self._libraries[library_id]
            return True
            
    # Document CRUD operations
    
    def add_document(self, library_id: UUID, document: Document) -> bool:
        """
        Add a document to a library.
        Raises NotFoundError if the library does not exist.
        Returns True if the document was added, False if it was updated.
        """
        with self._lock.write_lock():
            library = self._get_library_or_raise(library_id)
            
            # Create a new list of documents with the new/updated document
            existing_docs = [doc for doc in library.documents if doc.id != document.id]
            exists = len(existing_docs) != len(library.documents)
            
            # Create a new library with the updated documents list
            updated_library = Library(
                id=library.id,
                documents=existing_docs + [document],
                metadata=library.metadata
            )
            
            self._libraries[library_id] = updated_library
            return not exists
    
    def get_document(self, library_id: UUID, document_id: UUID) -> Document:
        """
        Get a document by ID from a library.
        Raises NotFoundError if the library or document does not exist.
        """
        with self._lock.read_lock():
            library = self._get_library_or_raise(library_id)
            for doc in library.documents:
                if doc.id == document_id:
                    return doc
            raise NotFoundError(f"Document with ID {document_id} not found in library {library_id}")
    
    def update_document(self, library_id: UUID, document_id: UUID, updated: Document) -> bool:
        """
        Update a document in a library.
        Raises NotFoundError if the library or document does not exist.
        Returns True if the document was updated.
        """
        with self._lock.write_lock():
            library = self._get_library_or_raise(library_id)
            
            # Check if the document exists
            if not any(doc.id == document_id for doc in library.documents):
                raise NotFoundError(f"Document with ID {document_id} not found in library {library_id}")
            
            # Create a new list of documents with the updated document
            updated_docs = [updated if doc.id == document_id else doc for doc in library.documents]
            
            # Create a new library with the updated documents list
            updated_library = Library(
                id=library.id,
                documents=updated_docs,
                metadata=library.metadata
            )
            
            self._libraries[library_id] = updated_library
            return True
    
    def delete_document(self, library_id: UUID, document_id: UUID) -> bool:
        """
        Delete a document from a library.
        Raises NotFoundError if the library or document does not exist.
        Returns True if the document was deleted.
        """
        with self._lock.write_lock():
            library = self._get_library_or_raise(library_id)
            
            # Check if the document exists
            if not any(doc.id == document_id for doc in library.documents):
                raise NotFoundError(f"Document with ID {document_id} not found in library {library_id}")
            
            # Create a new list of documents without the deleted document
            updated_docs = [doc for doc in library.documents if doc.id != document_id]
            
            # Create a new library with the updated documents list
            updated_library = Library(
                id=library.id,
                documents=updated_docs,
                metadata=library.metadata
            )
            
            self._libraries[library_id] = updated_library
            return True
    
    # Chunk CRUD operations
    
    def add_chunk(self, library_id: UUID, document_id: UUID, chunk: Chunk) -> bool:
        """
        Add a chunk to a document in a library.
        Raises NotFoundError if the library or document does not exist.
        Returns True if the chunk was added, False if it was updated.
        """
        with self._lock.write_lock():
            document = self._get_document_or_raise(library_id, document_id)
            
            # Create a new list of chunks with the new/updated chunk
            existing_chunks = [c for c in document.chunks if c.id != chunk.id]
            exists = len(existing_chunks) != len(document.chunks)
            
            # Create a new document with the updated chunks list
            updated_document = Document(
                id=document.id,
                chunks=existing_chunks + [chunk],
                metadata=document.metadata
            )
            
            # Update the document in the library
            self.update_document(library_id, document_id, updated_document)
            return not exists
    
    def get_chunk(self, library_id: UUID, document_id: UUID, chunk_id: UUID) -> Chunk:
        """
        Get a chunk by ID from a document in a library.
        Raises NotFoundError if the library, document, or chunk does not exist.
        """
        with self._lock.read_lock():
            document = self._get_document_or_raise(library_id, document_id)
            for chunk in document.chunks:
                if chunk.id == chunk_id:
                    return chunk
            raise NotFoundError(
                f"Chunk with ID {chunk_id} not found in document {document_id} in library {library_id}"
            )
    
    def update_chunk(self, library_id: UUID, document_id: UUID, chunk_id: UUID, updated: Chunk) -> bool:
        """
        Update a chunk in a document in a library.
        Raises NotFoundError if the library, document, or chunk does not exist.
        Returns True if the chunk was updated.
        """
        with self._lock.write_lock():
            document = self._get_document_or_raise(library_id, document_id)
            
            # Check if the chunk exists
            if not any(chunk.id == chunk_id for chunk in document.chunks):
                raise NotFoundError(
                    f"Chunk with ID {chunk_id} not found in document {document_id} in library {library_id}"
                )
            
            # Create a new list of chunks with the updated chunk
            updated_chunks = [updated if chunk.id == chunk_id else chunk for chunk in document.chunks]
            
            # Create a new document with the updated chunks list
            updated_document = Document(
                id=document.id,
                chunks=updated_chunks,
                metadata=document.metadata
            )
            
            # Update the document in the library
            self.update_document(library_id, document_id, updated_document)
            return True
    
    def delete_chunk(self, library_id: UUID, document_id: UUID, chunk_id: UUID) -> bool:
        """
        Delete a chunk from a document in a library.
        Raises NotFoundError if the library, document, or chunk does not exist.
        Returns True if the chunk was deleted.
        """
        with self._lock.write_lock():
            document = self._get_document_or_raise(library_id, document_id)
            
            # Check if the chunk exists
            if not any(chunk.id == chunk_id for chunk in document.chunks):
                raise NotFoundError(
                    f"Chunk with ID {chunk_id} not found in document {document_id} in library {library_id}"
                )
            
            # Create a new list of chunks without the deleted chunk
            updated_chunks = [chunk for chunk in document.chunks if chunk.id != chunk_id]
            
            # Create a new document with the updated chunks list
            updated_document = Document(
                id=document.id,
                chunks=updated_chunks,
                metadata=document.metadata
            )
            
            # Update the document in the library
            self.update_document(library_id, document_id, updated_document)
            return True
    
    # Helper methods
    
    def _get_library_or_raise(self, library_id: UUID) -> Library:
        """Get a library by ID or raise NotFoundError if it doesn't exist."""
        if library_id not in self._libraries:
            raise NotFoundError(f"Library with ID {library_id} not found")
        return self._libraries[library_id]
    
    def _get_document_or_raise(self, library_id: UUID, document_id: UUID) -> Document:
        """Get a document by ID from a library or raise NotFoundError if it doesn't exist."""
        library = self._get_library_or_raise(library_id)
        for doc in library.documents:
            if doc.id == document_id:
                return doc
        raise NotFoundError(f"Document with ID {document_id} not found in library {library_id}")
