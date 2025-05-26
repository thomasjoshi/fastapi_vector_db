"""
Embedding utilities for vector database.

This module provides functions to generate embeddings from text
using the Cohere API.
"""

from typing import List, Optional, cast

import cohere  # type: ignore[import-untyped]
from loguru import logger

from app.core.config import settings


class EmbeddingGenerator:
    """
    Utility class for generating embeddings from text using the Cohere API.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the embedding generator.

        Args:
            api_key: Cohere API key (defaults to settings.COHERE_API_KEY)
            model: Cohere embedding model to use (defaults to
                settings.COHERE_EMBEDDING_MODEL)
        """
        self.api_key = api_key or settings.COHERE_API_KEY
        self.model = model or settings.COHERE_EMBEDDING_MODEL
        self.client = cohere.Client(api_key=self.api_key)
        logger.info(f"Initialized Cohere embedding generator with model: {self.model}")

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using the Cohere API.

        Args:
            texts: List of text strings to generate embeddings for

        Returns:
            List of embedding vectors (empty list for each text if an error occurs)
        """
        if not texts:
            logger.debug("No texts provided for embedding generation")
            return []

        try:
            # Call Cohere API to generate embeddings
            # Note: The Cohere client is not async, so we're using it in a way that
            # won't block the event loop for this potentially blocking operation
            import asyncio

            response = await asyncio.to_thread(
                self.client.embed,
                texts=texts,
                model=self.model,
                input_type="search_document",
            )

            # Extract embeddings from response
            embeddings_data = response.embeddings
            # Assure mypy of the type from the untyped library
            typed_embeddings = cast(List[List[float]], embeddings_data)

            logger.info(
                f"Generated {len(typed_embeddings)} embeddings using model {self.model}"
            )
            return typed_embeddings
        except Exception as e:
            logger.error(f"Cohere API error generating embeddings: {str(e)}")
            # Return empty embeddings in case of error
            return [[] for _ in texts]

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text.

        Args:
            text: Text string to generate embedding for

        Returns:
            Embedding vector
        """
        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if embeddings else []


# Singleton instance
_embedding_generator = EmbeddingGenerator()


def get_embedding_generator() -> EmbeddingGenerator:
    """
    Get the singleton embedding generator instance.

    Returns:
        EmbeddingGenerator instance
    """
    return _embedding_generator
