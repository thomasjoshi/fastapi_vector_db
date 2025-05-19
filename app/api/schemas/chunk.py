"""
Schemas for chunk operations.
"""

from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ChunkCreate(BaseModel):
    """Schema for creating a new chunk."""
    
    text: str
    embedding: List[float]
    metadata: Dict[str, str] = Field(default_factory=dict)


class ChunkRead(BaseModel):
    """Schema for reading a chunk."""

    id: UUID
    text: str
    embedding: List[float]
    metadata: Dict[str, str] = Field(default_factory=dict)

    class Config:
        """Pydantic config."""

        from_attributes = True


class SearchQuery(BaseModel):
    """Schema for search queries."""

    embedding: List[float]
    k: int = Field(default=5, ge=1, le=100)


class SearchHit(BaseModel):
    """Schema for a single search result."""
    
    chunk: ChunkRead
    score: float


class SearchResponse(BaseModel):
    """Schema for search response."""
    
    hits: List[SearchHit]


class IndexResponse(BaseModel):
    """Schema for index operation responses."""

    chunks_indexed: int
