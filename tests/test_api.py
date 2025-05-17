"""Tests for the libraries API."""

from fastapi import status
from fastapi.testclient import TestClient

from app.domain.models import Library
from app.main import app

# Create a test client
client = TestClient(app)


def test_create_library() -> None:
    """Test creating a library via the API with Location header."""
    # Create a library without ID (using schema)
    lib = Library.example()
    # Use model_dump to properly serialize the model
    lib_data = {"documents": lib.model_dump()["documents"], "metadata": lib.metadata}
    response = client.post("/libraries/", json=lib_data)

    # Check response
    assert response.status_code == status.HTTP_201_CREATED
    library_id = response.json()
    assert library_id is not None

    # Check Location header
    assert "Location" in response.headers
    assert response.headers["Location"] == f"/libraries/{library_id}"


def test_get_library() -> None:
    """Test getting a library via the API with correct schema."""
    # Create a library without ID (using schema)
    lib = Library.example()
    # Use model_dump to properly serialize the model
    lib_data = {"documents": lib.model_dump()["documents"], "metadata": lib.metadata}
    response = client.post("/libraries/", json=lib_data)
    library_id = response.json()

    # Get the library
    response = client.get(f"/libraries/{library_id}")

    # Check response
    assert response.status_code == status.HTTP_200_OK
    retrieved_lib = response.json()
    assert retrieved_lib["id"] == library_id
    assert len(retrieved_lib["documents"]) == len(lib.documents)

    # Verify schema fields
    assert "id" in retrieved_lib
    assert "documents" in retrieved_lib
    assert "metadata" in retrieved_lib


def test_update_library() -> None:
    """Test updating a library via the API with full data."""
    # Create a library without ID (using schema)
    lib = Library.example()
    # Use model_dump to properly serialize the model
    lib_data = {"documents": lib.model_dump()["documents"], "metadata": lib.metadata}
    response = client.post("/libraries/", json=lib_data)
    library_id = response.json()

    # Update the library with full data
    updated_data = {
        "documents": lib.model_dump()["documents"],
        "metadata": {"name": "Updated via API", "created_by": "Test"},
    }
    response = client.put(f"/libraries/{library_id}", json=updated_data)

    # Check response
    assert response.status_code == status.HTTP_204_NO_CONTENT

    # Get the updated library
    response = client.get(f"/libraries/{library_id}")
    retrieved_lib = response.json()
    assert retrieved_lib["metadata"]["name"] == "Updated via API"


def test_update_library_with_partial_data() -> None:
    """Test updating a library via the API with partial data."""
    # Create a library without ID (using schema)
    lib = Library.example()
    # Use model_dump to properly serialize the model
    lib_data = {"documents": lib.model_dump()["documents"], "metadata": lib.metadata}
    response = client.post("/libraries/", json=lib_data)
    library_id = response.json()

    # Update only the metadata
    update_data = {"metadata": {"name": "Updated via API", "created_by": "Test"}}
    response = client.put(f"/libraries/{library_id}", json=update_data)

    # Check response
    assert response.status_code == status.HTTP_204_NO_CONTENT

    # Get the updated library
    response = client.get(f"/libraries/{library_id}")
    retrieved_lib = response.json()
    assert retrieved_lib["metadata"]["name"] == "Updated via API"
    # Documents should remain unchanged
    assert len(retrieved_lib["documents"]) == len(lib.documents)


def test_delete_library() -> None:
    """Test deleting a library via the API."""
    # Create a library without ID (using schema)
    lib = Library.example()
    # Use model_dump to properly serialize the model
    lib_data = {"documents": lib.model_dump()["documents"], "metadata": lib.metadata}
    response = client.post("/libraries/", json=lib_data)
    library_id = response.json()

    # Delete the library
    response = client.delete(f"/libraries/{library_id}")

    # Check response
    assert response.status_code == status.HTTP_204_NO_CONTENT

    # Verify deletion
    response = client.get(f"/libraries/{library_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_delete_library_not_found() -> None:
    """Test deleting a non-existent library via the API."""
    # Try to delete a non-existent library
    response = client.delete("/libraries/00000000-0000-0000-0000-000000000000")

    # Check response
    assert response.status_code == status.HTTP_404_NOT_FOUND
    error_data = response.json()

    # Verify error schema
    assert "detail" in error_data
    assert "resource_type" in error_data
    assert "resource_id" in error_data
    assert error_data["resource_type"] == "Library"


def test_centralized_error_handling() -> None:
    """Test that centralized error handling works for NotFoundError."""
    # Try to get a non-existent library
    response = client.get("/libraries/00000000-0000-0000-0000-000000000000")

    # Check response
    assert response.status_code == status.HTTP_404_NOT_FOUND
    error_data = response.json()

    # Verify error schema from centralized handler
    assert "detail" in error_data
    assert "resource_type" in error_data
    assert "resource_id" in error_data
    assert error_data["resource_type"] == "Library"
    assert error_data["resource_id"] == "00000000-0000-0000-0000-000000000000"
