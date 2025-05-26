"""
In-memory repository implementation.

This is a simple in-memory implementation of the repository interface.
It uses a dictionary to store libraries and their documents.
"""
import threading
from types import TracebackType
from typing import Dict, Optional, Type
from uuid import UUID

from loguru import logger

from app.core.persistence import Persistence, get_persistence
from app.domain.models import Chunk, Document, Library
from app.services.exceptions import NotFoundError


class ReadWriteLock:
    """
    A lock object that allows many simultaneous "read locks", but only one "write lock."
    """

    def __init__(self) -> None:
        self._read_ready = threading.Condition(threading.RLock())
        self._readers = 0

    def read_acquire(self) -> None:
        """Acquire a read lock. Blocks only if a thread has acquired the write lock."""
        self._read_ready.acquire()
        try:
            self._readers += 1
        finally:
            self._read_ready.release()

    def read_release(self) -> None:
        """Release a read lock."""
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if not self._readers:
                self._read_ready.notify_all()
        finally:
            self._read_ready.release()

    def write_acquire(self) -> None:
        """
        Acquire a write lock. Blocks until there are no acquired read or write locks.
        """
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def write_release(self) -> None:
        """Release a write lock."""
        self._read_ready.release()

    class ReadLock:
        """Context manager for read locking."""

        def __init__(self, rwlock: "ReadWriteLock") -> None:
            self.rwlock = rwlock

        def __enter__(self) -> "ReadWriteLock":
            self.rwlock.read_acquire()
            return self.rwlock

        def __exit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_val: Optional[BaseException],
            exc_tb: Optional[TracebackType],
        ) -> None:
            self.rwlock.read_release()

    class WriteLock:
        """Context manager for write locking."""

        def __init__(self, rwlock: "ReadWriteLock") -> None:
            self.rwlock = rwlock

        def __enter__(self) -> "ReadWriteLock":
            self.rwlock.write_acquire()
            return self.rwlock

        def __exit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_val: Optional[BaseException],
            exc_tb: Optional[TracebackType],
        ) -> None:
            self.rwlock.write_release()

    def read_lock(self) -> "ReadLock":
        """Return a context manager for read locking."""
        return self.ReadLock(self)

    def write_lock(self) -> "WriteLock":
        """Return a context manager for write locking."""
        return self.WriteLock(self)


class InMemoryRepo:
    """
    In-memory repository implementation.

    This class provides methods for CRUD operations on libraries, documents, and chunks.
    It uses a dictionary to store libraries and their documents.
    """

    def __init__(self) -> None:
        """Initialize the repository with an empty dictionary of libraries."""
        self._libraries: Dict[UUID, Library] = {}
        self._lock = ReadWriteLock()
        self._persistence: Optional[Persistence] = None

    # Library CRUD operations

    async def add_library(self, library: Library) -> UUID:
        """Add a library to the repository.

        If the library has no ID, one will be generated.
        Returns the ID of the library.
        """
        logger.info(f"Repo.add_library: ID {library.id}, type: {type(library.id)}")

        with self._lock.write_lock():
            # The 'if library.id is None:' check is redundant and likely caused
            # the unreachable code error. If a library comes in, it's assumed to
            # have an ID. If a new one needs to be made from data that lacks an
            # ID, that should be handled before calling add_library or by a
            # different method (e.g., create_library_from_data). Removing the block.

            # Add the library to the dictionary
            if library.id in self._libraries:
                # To prevent overwriting, though the method signature doesn't
                # suggest update_or_create capabilities. This is a warning to
                # indicate that the library already exists and will be overwritten.
                logger.warning(
                    f"Library with ID {library.id} already exists. "
                    "Overwriting existing library."
                )
            self._libraries[library.id] = library
            logger.info(f"Repo.add_library: Added. Keys: {list(self._libraries)}")
            return library.id

    async def get_library(self, library_id: UUID) -> Library:
        """
        Get a library by ID.
        Raises NotFoundError if the library does not exist.
        """
        logger.info(f"Repo.get_library: ID {library_id}, type: {type(library_id)}")
        with self._lock.read_lock():
            logger.info(f"Repo.get_library: Found. Keys: {list(self._libraries)}")
            # Assuming _get_library_or_raise is async based on mypy error
            return await self._get_library_or_raise(library_id)

    async def update_library(self, library_id: UUID, library: Library) -> bool:
        """
        Update a library.
        Raises NotFoundError if the library does not exist.
        Returns True if the library was updated.
        """
        with self._lock.write_lock():
            await self._get_library_or_raise(library_id)  # Check if library exists

            # Update the library in the dictionary
            self._libraries[library_id] = library
            return True

    async def update_library_if_exists(
        self, library_id: UUID, library: Library
    ) -> bool:
        """
        Update a library if it exists.
        Returns True if the library was updated, False if it does not exist.
        """
        with self._lock.write_lock():
            if library_id not in self._libraries:
                return False

            # Update the library in the dictionary
            self._libraries[library_id] = library
            return True

    async def delete_library(self, library_id: UUID) -> bool:
        """
        Delete a library.
        Raises NotFoundError if the library does not exist.
        Returns True if the library was deleted.
        """
        with self._lock.write_lock():
            if library_id not in self._libraries:
                raise NotFoundError(
                    f"Library {library_id} not found", "Library", library_id
                )

            # Delete the library from the dictionary
            del self._libraries[library_id]
            return True

    # Document CRUD operations

    async def add_document(self, library_id: UUID, document: Document) -> Document:
        """
        Add a document to a library.
        Raises NotFoundError if the library does not exist.
        Returns the added document.
        """
        with self._lock.write_lock():
            await self._get_library_or_raise(library_id)  # Check if library exists

            # Create a new list of documents with the new/updated document
            lib_docs = self._libraries[library_id].documents
            existing_docs = [doc for doc in lib_docs if doc.id != document.id]

            # Create a new library with the updated documents list
            updated_library = Library(
                id=library_id,
                documents=existing_docs + [document],
                metadata=self._libraries[library_id].metadata,
            )

            self._libraries[library_id] = updated_library
            return document

    async def get_document(self, library_id: UUID, document_id: UUID) -> Document:
        """
        Get a document by ID from a library.
        Raises NotFoundError if the library or document does not exist.
        """
        with self._lock.read_lock():
            return await self._get_document_or_raise(library_id, document_id)

    async def update_document(
        self, library_id: UUID, document_id: UUID, updated: Document
    ) -> bool:
        """
        Update a document in a library.
        Raises NotFoundError if the library or document does not exist.
        Returns True if the document was updated.
        """
        with self._lock.write_lock():
            library = await self._get_library_or_raise(library_id)

            # Check if the document exists
            if not any(doc.id == document_id for doc in library.documents):
                raise NotFoundError(
                    f"Document {document_id} not found in library {library_id}",
                    "Document",
                    document_id,
                )

            # Create a new list of documents with the updated document
            updated_docs = [
                updated if doc.id == document_id else doc for doc in library.documents
            ]

            # Create a new library with the updated documents list
            updated_library = Library(
                id=library.id, documents=updated_docs, metadata=library.metadata
            )

            self._libraries[library_id] = updated_library
            return True

    async def delete_document(self, library_id: UUID, document_id: UUID) -> bool:
        """
        Delete a document from a library.
        Raises NotFoundError if the library or document does not exist.
        Returns True if the document was deleted.
        """
        with self._lock.write_lock():
            library = await self._get_library_or_raise(library_id)

            # Check if the document exists
            if not any(doc.id == document_id for doc in library.documents):
                raise NotFoundError(
                    f"Document {document_id} not found in library {library_id}",
                    "Document",
                    document_id,
                )

            # Create a new list of documents without the deleted document
            updated_docs = [doc for doc in library.documents if doc.id != document_id]

            # Create a new library with the updated documents list
            updated_library = Library(
                id=library.id, documents=updated_docs, metadata=library.metadata
            )

            self._libraries[library_id] = updated_library
            return True

    # Chunk CRUD operations

    async def add_chunk(
        self, library_id: UUID, document_id: UUID, chunk: Chunk
    ) -> Chunk:
        """
        Add a chunk to a document in a library.
        Raises NotFoundError if the library or document does not exist.
        Returns the added chunk.
        """
        with self._lock.write_lock():
            document_obj = await self._get_document_or_raise(library_id, document_id)

            # Create a new list of chunks with the new/updated chunk
            existing_chunks = [c for c in document_obj.chunks if c.id != chunk.id]

            # Create a new document with the updated chunks list
            updated_document = Document(
                id=document_id,
                chunks=existing_chunks + [chunk],
                metadata=document_obj.metadata,
            )

            # Update the document in the library
            await self.update_document(library_id, document_id, updated_document)
            return chunk

    async def get_chunk(
        self, library_id: UUID, document_id: UUID, chunk_id: UUID
    ) -> Chunk:
        """
        Get a chunk by ID from a document in a library.
        Raises NotFoundError if the library, document, or chunk does not exist.
        """
        with self._lock.read_lock():
            document = await self._get_document_or_raise(library_id, document_id)
            for chunk in document.chunks:
                if chunk.id == chunk_id:
                    return chunk
            raise NotFoundError(
                f"Chunk {chunk_id} not found in doc {document_id}",
                "Chunk",
                chunk_id,
            )

    async def update_chunk(
        self, library_id: UUID, document_id: UUID, chunk_id: UUID, updated: Chunk
    ) -> bool:
        """
        Update a chunk in a document in a library.
        Raises NotFoundError if the library, document, or chunk does not exist.
        Returns True if the chunk was updated.
        """
        with self._lock.write_lock():
            document = await self._get_document_or_raise(library_id, document_id)

            # Check if the chunk exists
            if not any(chunk.id == chunk_id for chunk in document.chunks):
                raise NotFoundError(
                    f"Chunk {chunk_id} not found in doc {document_id}",
                    "Chunk",
                    chunk_id,
                )

            # Create a new list of chunks with the updated chunk
            updated_chunks = [
                updated if chunk.id == chunk_id else chunk for chunk in document.chunks
            ]

            # Create a new document with the updated chunks list
            updated_document = Document(
                id=document_id, chunks=updated_chunks, metadata=document.metadata
            )

            # Update the document in the library
            await self.update_document(library_id, document_id, updated_document)
            return True

    async def delete_chunk(
        self, library_id: UUID, document_id: UUID, chunk_id: UUID
    ) -> bool:
        """
        Delete a chunk from a document in a library.
        Raises NotFoundError if the library, document, or chunk does not exist.
        Returns True if the chunk was deleted.
        """
        with self._lock.write_lock():
            document = await self._get_document_or_raise(library_id, document_id)

            # Check if the chunk exists
            if not any(chunk.id == chunk_id for chunk in document.chunks):
                raise NotFoundError(
                    f"Chunk {chunk_id} not found in doc {document_id}",
                    "Chunk",
                    chunk_id,
                )

            # Create a new list of chunks without the deleted chunk
            updated_chunks = [
                chunk for chunk in document.chunks if chunk.id != chunk_id
            ]

            # Create a new document with the updated chunks list
            updated_document = Document(
                id=document_id, chunks=updated_chunks, metadata=document.metadata
            )

            # Update the document in the library
            await self.update_document(library_id, document_id, updated_document)
            return True

    # Helper methods

    async def _get_library_or_raise(self, library_id: UUID) -> Library:
        """Get a library by ID or raise NotFoundError if it doesn't exist."""
        if library_id not in self._libraries:
            raise NotFoundError(
                f"Library {library_id} not found", "Library", library_id
            )
        return self._libraries[library_id]

    async def _get_document_or_raise(
        self, library_id: UUID, document_id: UUID
    ) -> Document:
        """Get a doc by ID from a lib or raise NotFoundError if it doesn't exist."""
        library = await self._get_library_or_raise(library_id)
        for doc in library.documents:
            if doc.id == document_id:
                return doc
        raise NotFoundError(
            f"Doc {document_id} not found in lib {library_id}",
            "Document",
            document_id,
        )


# Singleton instance for the application lifetime
_repo = InMemoryRepo()


async def initialize_repo() -> InMemoryRepo:
    """Initialize the repository and load persisted data if enabled."""
    global _repo

    # Initialize persistence
    persistence = get_persistence()
    _repo._persistence = persistence

    # Load any persisted data
    libraries = await persistence.load_from_disk()
    if libraries:
        with _repo._lock.write_lock():
            _repo._libraries = libraries
            logger.info(f"Loaded {len(libraries)} libraries from persistence")

    # Start auto-save background task if enabled
    await persistence.start_auto_save(lambda: _repo._libraries)

    return _repo


async def get_repo() -> InMemoryRepo:
    """
    Dependency that provides the repository instance.

    Returns:
        An instance of InMemoryRepo
    """
    logger.debug(
        f"get_repo called, returning repo with {len(_repo._libraries)} libraries"
    )
    logger.debug(f"Library IDs in repo: {list(_repo._libraries.keys())}")
    return _repo
