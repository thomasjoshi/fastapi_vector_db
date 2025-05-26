from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.domain.models import Library


def test_library_example() -> None:
    # Test that Library.example() returns a valid instance
    lib = Library.example()
    assert len(lib.documents) > 0

    # Test immutability by attempting to modify the id
    with pytest.raises(ValidationError):
        lib.id = uuid4()  # type: ignore[misc]
