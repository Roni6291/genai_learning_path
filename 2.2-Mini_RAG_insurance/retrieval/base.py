from abc import ABC, abstractmethod

import numpy as np
from loguru import logger


class BaseVectorStore(ABC):
    """Base class for vector stores."""

    @property
    @abstractmethod
    def texts(self) -> list[str] | None:
        """Get the stored texts.

        Returns:
            list[str] | None: Stored texts or None if not set
        """
        pass

    @property
    @abstractmethod
    def vectors(self) -> list[np.ndarray] | None:
        """Get the stored vectors.

        Returns:
            list[np.ndarray] | None: Stored vectors or None if not set
        """
        pass

    @abstractmethod
    def vectorize(self, texts: list[str]) -> None:
        """Fit the vectorizer and transform the stored texts.

        Args:
            texts: Text data to vectorize
        """
        pass

    @abstractmethod
    def as_retriever(self) -> "BaseRetriever":
        """Get a Retriever instance using this vector store.

        Returns:
            BaseRetriever: Retriever instance
        """
        pass


class BaseRetriever(ABC):
    """Base class for retrievers."""

    def __init__(self, vector_store: "BaseVectorStore"):
        """Initialize the BaseRetriever.

        Args:
            vector_store: An instance of BaseVectorStore

        Raises:
            RuntimeError: If vector_store is not fitted or transformed and stored
        """
        logger.debug("Initializing BaseRetriever")

        if vector_store.texts is None or vector_store.vectors is None:
            logger.error(
                "Vector store must have texts and vectors before creating retriever"
            )
            raise RuntimeError(
                "Vector store does not have texts and vectors. Call vectorize() first."
            )

        self.vector_store = vector_store
        logger.info("BaseRetriever initialized")

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Retrieve the most relevant texts for a given query.

        Args:
            query: The query string
            top_k: Number of top relevant texts to retrieve

        Returns:
            list[tuple[str, float]]: List of tuples containing the text and its similarity score
        """
        pass
