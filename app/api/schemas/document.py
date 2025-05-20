from uuid import UUID

from pydantic import BaseModel


class DocumentCreate(BaseModel):
    """Schema for creating a new document."""

    metadata: dict[str, str] = {}


class DocumentRead(DocumentCreate):
    """Schema for reading a document."""

    id: UUID
