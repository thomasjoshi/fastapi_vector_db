"""
Persistence implementation for the Vector Database.

This module provides functionality to save and load the database state
to/from disk using JSON serialization.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from uuid import UUID, uuid4

from loguru import logger
from pydantic import BaseModel

from app.core.config import settings
from app.domain.models import Chunk, Document, Library


class JSONEncoder(json.JSONEncoder):
    """Custom JSON Encoder to handle UUID and datetime objects."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        return super().default(obj)


class Persistence:
    """
    Handles saving and loading the database state to/from disk.
    """

    def __init__(self, persistence_path: Optional[str] = None):
        """
        Initialize the persistence manager.

        Args:
            persistence_path: Path to save data. If None, uses config setting.
        """
        self.persistence_path = (
            persistence_path or settings.PERSISTENCE_PATH or "./data/vector_db.json"
        )
        self.persistence_enabled = settings.ENABLE_PERSISTENCE
        self.persistence_interval = settings.PERSISTENCE_INTERVAL
        self._save_task: Optional[asyncio.Task[None]] = None

    async def save_to_disk(self, libraries: Dict[UUID, Library]) -> bool:
        """
        Save the current state of libraries to disk as JSON.

        Args:
            libraries: Dictionary of libraries to save.

        Returns:
            True if save was successful, False otherwise.
        """
        if not self.persistence_enabled:
            logger.debug("Persistence is disabled, skipping save.")
            return False

        try:
            # Ensure the directory exists
            path = Path(self.persistence_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Convert libraries to a serializable format
            data_to_save = {str(k): v.model_dump() for k, v in libraries.items()}

            # Write to a temporary file first
            temp_path = f"{self.persistence_path}.tmp"
            with open(temp_path, "w") as f:
                json.dump(data_to_save, f, cls=JSONEncoder, indent=2)

            # Rename to the actual file (atomic operation on most filesystems)
            os.replace(temp_path, self.persistence_path)

            logger.info(f"Saved database to {self.persistence_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving database to disk: {e}")
            return False

    async def load_from_disk(self) -> Dict[UUID, Library]:
        """
        Load libraries from disk.

        Returns:
            Dict of loaded libraries from disk, or empty dict if no file.
        """
        if not self.persistence_enabled:
            logger.debug("Persistence is disabled, skipping load.")
            return {}

        try:
            path = Path(self.persistence_path)
            if not path.exists():
                logger.info(f"No persistence file found at {self.persistence_path}")
                return {}

            with open(path) as f:
                data = json.load(f)

            # Convert the loaded data back to Library objects
            libraries: Dict[UUID, Library] = {}
            for lib_id, lib_data in data.items():
                # Convert documents data
                documents = []
                for doc_data in lib_data.get("documents", []):
                    # Convert chunks data
                    chunks = []
                    for chunk_data in doc_data.get("chunks", []):
                        chunk = Chunk(
                            id=(
                                UUID(chunk_data["id"])
                                if "id" in chunk_data
                                else uuid4()
                            ),
                            document_id=(
                                UUID(doc_data["id"])
                            ),
                            text=chunk_data["text"],
                            embedding=chunk_data["embedding"],
                            metadata=chunk_data.get("metadata", {}),
                        )
                        chunks.append(chunk)

                    document = Document(
                        id=UUID(doc_data["id"]) if "id" in doc_data else uuid4(),
                        chunks=chunks,
                        metadata=doc_data.get("metadata", {}),
                    )
                    documents.append(document)

                # Create the library
                library = Library(
                    id=UUID(lib_id),
                    documents=documents,
                    metadata=lib_data.get("metadata", {}),
                )
                libraries[UUID(lib_id)] = library

            logger.info(f"Loaded {len(libraries)} libs from {self.persistence_path}")
            return libraries
        except Exception as e:
            logger.error(f"Error loading database from disk: {e}")
            return {}

    async def start_auto_save(
        self, get_libraries_func: Callable[[], Dict[UUID, Library]]
    ) -> None:
        """
        Start the auto-save background task.

        Args:
            get_libraries_func: Function that returns the current libraries dictionary.
        """
        if not self.persistence_enabled or self.persistence_interval <= 0:
            logger.info("Auto-save is disabled")
            return

        async def auto_save_task() -> None:
            try:
                while True:
                    await asyncio.sleep(self.persistence_interval)
                    libraries = get_libraries_func()
                    await self.save_to_disk(libraries)
                    logger.debug(f"Auto-saved database with {len(libraries)} libraries")
            except asyncio.CancelledError:
                logger.info("Auto-save task cancelled")
                # Save one last time before exiting
                libraries = get_libraries_func()
                await self.save_to_disk(libraries)
            except Exception as e:
                logger.error(f"Error in auto-save task: {e}")

        self._save_task = asyncio.create_task(auto_save_task())
        logger.info(f"Auto-save task started, interval: {self.persistence_interval}s")

    async def stop_auto_save(self) -> None:
        """Stop the auto-save background task."""
        if self._save_task:
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass
            self._save_task = None
            logger.info("Stopped auto-save task")


# Singleton instance
_persistence = Persistence()


def get_persistence() -> Persistence:
    """
    Get the singleton persistence instance.

    Returns:
        The Persistence instance.
    """
    return _persistence
