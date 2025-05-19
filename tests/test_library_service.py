"""
Unit tests for the LibraryService class.
"""
from typing import List, Tuple
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock

from app.domain.models import Library
from app.repos.in_memory import InMemoryRepo
from app.repos.in_memory import NotFoundError as RepoNotFoundError
from app.services.exceptions import NotFoundError
from app.services.library import LibraryService


class TestLibraryService:
    """Tests for the LibraryService class."""

    @pytest_asyncio.fixture
    async def mock_repo(self):
        """Create a mock repository."""
        mock = AsyncMock(spec=InMemoryRepo)
        return mock

    @pytest.fixture
    def mock_metrics(self):
        """Create a mock metrics callback."""
        return MagicMock()

    @pytest.fixture
    def metrics_calls(self) -> List[Tuple[str, dict]]:
        """Track metrics calls."""
        calls = []

        def metrics_callback(metric_name: str, **kwargs):
            calls.append((metric_name, kwargs))

        return calls, metrics_callback

    @pytest.mark.asyncio
    async def test_add_library(self, mock_repo, mock_metrics):
        """Test adding a library."""
        # Arrange
        service = LibraryService(mock_repo, mock_metrics)
        lib = Library.example()
        mock_repo.add_library.return_value = lib.id

        # Act
        result = await service.add_library(lib)

        # Assert
        assert result == lib.id
        mock_repo.add_library.assert_called_once_with(lib)
        mock_metrics.assert_called_once_with("library.add", library_id=str(lib.id))

    @pytest.mark.asyncio
    async def test_get_library(self, mock_repo, mock_metrics):
        """Test getting a library."""
        # Arrange
        service = LibraryService(mock_repo, mock_metrics)
        lib = Library.example()
        lib_id = lib.id
        mock_repo.get_library.return_value = lib

        # Act
        result = await service.get_library(lib_id)

        # Assert
        assert result == lib
        mock_repo.get_library.assert_called_once_with(lib_id)
        mock_metrics.assert_called_once_with("library.get", library_id=str(lib_id))

    @pytest.mark.asyncio
    async def test_get_library_not_found(self, mock_repo, mock_metrics):
        """Test getting a non-existent library."""
        # Arrange
        service = LibraryService(mock_repo, mock_metrics)
        lib_id = uuid4()
        mock_repo.get_library.side_effect = RepoNotFoundError(f"Library with ID {lib_id} not found")

        # Act & Assert
        with pytest.raises(NotFoundError) as excinfo:
            await service.get_library(lib_id)

        # Verify error details
        assert "Library" in str(excinfo.value)
        assert str(lib_id) in str(excinfo.value)
        mock_repo.get_library.assert_called_once_with(lib_id)
        mock_metrics.assert_called_once_with("library.get", library_id=str(lib_id))

    @pytest.mark.asyncio
    async def test_update_library_atomic(self, mock_repo, mock_metrics):
        """Test updating a library with atomic operation."""
        # Arrange
        service = LibraryService(mock_repo, mock_metrics)
        lib = Library.example()
        lib_id = lib.id
        mock_repo.update_library_if_exists.return_value = True

        # Act
        await service.update_library(lib_id, lib)

        # Assert
        mock_repo.update_library_if_exists.assert_called_once_with(lib_id, lib)
        mock_metrics.assert_called_once_with("library.update", library_id=str(lib_id))
        # Ensure we didn't need to fall back to the existence check
        mock_repo.get_library.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_library_not_found(self, mock_repo, mock_metrics):
        """Test updating a non-existent library."""
        # Arrange
        service = LibraryService(mock_repo, mock_metrics)
        lib = Library.example()
        lib_id = lib.id
        
        # Set up update_library_if_exists to return False
        mock_repo.update_library_if_exists.return_value = False
        
        # Set up get_library to raise NotFoundError
        mock_repo.get_library.side_effect = RepoNotFoundError(f"Library with ID {lib_id} not found")

        # Act & Assert
        with pytest.raises(NotFoundError) as excinfo:
            await service.update_library(lib_id, lib)

        # Verify error details
        assert "Library" in str(excinfo.value)
        assert str(lib_id) in str(excinfo.value)
        mock_repo.update_library_if_exists.assert_called_once_with(lib_id, lib)
        mock_repo.get_library.assert_called_once_with(lib_id)
        mock_metrics.assert_called_once_with("library.update", library_id=str(lib_id))

    @pytest.mark.asyncio
    async def test_delete_library(self, mock_repo, mock_metrics):
        """Test deleting a library."""
        # Arrange
        service = LibraryService(mock_repo, mock_metrics)
        lib_id = uuid4()
        mock_repo.delete_library.return_value = True

        # Act
        await service.delete_library(lib_id)

        # Assert
        mock_repo.delete_library.assert_called_once_with(lib_id)
        mock_metrics.assert_called_once_with("library.delete", library_id=str(lib_id))

    @pytest.mark.asyncio
    async def test_delete_library_not_found(self, mock_repo, mock_metrics):
        """Test deleting a non-existent library."""
        # Arrange
        service = LibraryService(mock_repo, mock_metrics)
        lib_id = uuid4()
        
        # Configure delete_library to raise error
        mock_repo.delete_library.side_effect = RepoNotFoundError(f"Library with ID {lib_id} not found")

        # Act & Assert
        with pytest.raises(NotFoundError) as excinfo:
            await service.delete_library(lib_id)

        # Verify error details
        assert "Library" in str(excinfo.value)
        assert str(lib_id) in str(excinfo.value)
        mock_repo.delete_library.assert_called_once_with(lib_id)
        mock_metrics.assert_called_once_with("library.delete", library_id=str(lib_id))

    @pytest.mark.asyncio
    async def test_metrics_callback_integration(self, metrics_calls):
        """Test that metrics callback is correctly invoked."""
        # Arrange
        calls, callback = metrics_calls
        repo = InMemoryRepo()
        service = LibraryService(repo, callback)
        lib = Library.example()

        # Act
        await service.add_library(lib)

        # Assert
        assert len(calls) == 1
        assert calls[0][0] == "library.add"
        assert calls[0][1]["library_id"] == str(lib.id)
