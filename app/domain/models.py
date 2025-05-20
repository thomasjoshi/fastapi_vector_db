import random
from typing import Any, Dict, List
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_serializer

__all__ = ["Chunk", "Document", "Library"]


class Chunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    text: str
    embedding: List[float]
    metadata: Dict[str, str] = {}

    model_config = {"frozen": True}

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        return {
            "id": str(self.id).replace('-', ''),  # Format UUID without dashes for consistent parsing
            "text": self.text,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }


class Document(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    chunks: List[Chunk]
    metadata: Dict[str, str] = {}

    model_config = {"frozen": True}

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        return {
            "id": str(self.id).replace('-', ''),  # Format UUID without dashes for consistent parsing
            "chunks": self.chunks,
            "metadata": self.metadata,
        }


class Library(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    documents: List[Document]
    metadata: Dict[str, str] = {}

    model_config = {"frozen": True}

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        return {
            "id": str(self.id).replace('-', ''),  # Format UUID without dashes for consistent parsing
            "documents": self.documents,
            "metadata": self.metadata,
        }

    @classmethod
    def example(cls) -> "Library":
        # Create a small example library with sample data
        chunks1 = [
            Chunk(
                text="This is the first chunk of document 1.",
                embedding=[random.random() for _ in range(5)],
                metadata={"source": "example", "page": "1"},
            ),
            Chunk(
                text="This is the second chunk of document 1.",
                embedding=[random.random() for _ in range(5)],
                metadata={"source": "example", "page": "1"},
            ),
        ]

        chunks2 = [
            Chunk(
                text="This is the first chunk of document 2.",
                embedding=[random.random() for _ in range(5)],
                metadata={"source": "example", "page": "2"},
            )
        ]

        documents = [
            Document(
                chunks=chunks1,
                metadata={"title": "Document 1", "author": "AI Assistant"},
            ),
            Document(
                chunks=chunks2,
                metadata={"title": "Document 2", "author": "AI Assistant"},
            ),
        ]

        return cls(
            documents=documents,
            metadata={"name": "Example Library", "created_by": "AI Assistant"},
        )


if __name__ == "__main__":
    # When run directly, print an example library
    lib = Library.example()
    print(lib)
