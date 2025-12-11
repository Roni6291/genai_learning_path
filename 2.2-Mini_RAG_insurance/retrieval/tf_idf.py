import sys

import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRetriever, BaseVectorStore


class TFIDFVectorStore(BaseVectorStore):
    def __init__(
        self,
        max_features: int | None = None,
        min_df: int = 1,
        max_df: float = 1.0,
    ):
        """Initialize the TFIDFVectorStore.

        This is an inmemory store that uses TF-IDF vectorization to
        convert text data into feature vectors for retrieval.

        Args:
            max_features: Maximum number of features (words) to extract.
                Default is 5000.
            min_df: Minimum document frequency for a word to be included.
                Words appearing in fewer documents will be ignored. Default is 2.
            max_df: Maximum document frequency (as proportion). Words appearing
                in more than this proportion of documents will be ignored.
        """
        logger.debug("Initializing TFIDFVectorStore")
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
        )

        self._is_fitted = False
        self._is_transformed = False
        self._texts = None
        self._vectors = None
        self.feature_names_ = None
        logger.info(
            f"TFIDFVectorStore initialized with max_features={max_features}, "
            f"min_df={min_df}, max_df={max_df}"
        )

    @property
    def is_fitted(self) -> bool:
        """Check if the vectorizer is fitted.

        Returns:
            bool: True if fitted, False otherwise
        """
        return self._is_fitted

    @property
    def is_transformed(self) -> bool:
        """Check if the vectorizer has transformed all texts.

        Returns:
            bool: True if transformed, False otherwise
        """
        return self._is_transformed

    @property
    def texts(self) -> None | list[str]:
        """Get the stored texts.

        Returns:
            list[str]: List of stored texts
        """
        if not self.is_transformed:
            return
        return self._texts

    @property
    def vectors(self) -> None | list[np.ndarray]:
        """Get the stored vectors.

        Returns:
            list[np.ndarray]: List of stored vectors
        """
        if not self.is_transformed:
            return
        return self._vectors

    def fit(self, texts: list[str]) -> "TFIDFVectorStore":
        """Fit the TF-IDF vectorizer on the sentences.

        Args:
            texts: Text data to fit the vectorizer on

        Returns:
            self: Returns the instance for method chaining
        """
        logger.info(f"Fitting vectorizer on {len(texts)} samples")

        self.vectorizer.fit(texts)
        self._is_fitted = True
        self.feature_names_ = self.vectorizer.get_feature_names_out()

        logger.info(f"Vectorizer fitted. Generated {len(self.feature_names_)} features")
        return self

    def transform(self, text: str) -> np.ndarray:
        """Transform text data into TF-IDF features.

        Args:
            texts: Text data to transform

        Returns:
            np.ndarray: TF-IDF feature matrix

        Raises:
            RuntimeError: If vectorizer hasn't been fitted yet
        """
        if not self.is_fitted:
            logger.error("Vectorizer must be fitted before transform")
            raise RuntimeError("Vectorizer not fitted. Call fit() first.")

        logger.debug(f"Transforming {text}")
        return self.vectorizer.transform([text]).toarray().flatten()  # type: ignore

    def transform_texts(
        self,
        texts: list[str],
    ) -> None:
        """transform the text data and save in memory.

        Args:
            texts: Text data to fit, transform and store in memory

        Returns:
            None
        """

        self._texts = texts
        self._vectors = [self.transform(text) for text in texts]

        logger.success(f"Stored {len(texts)} vectors in memory")
        self._is_transformed = True

    def vectorize(self, texts: list[str]) -> None:
        """Fit the vectorizer and transform the stored texts.

        Args:
            texts: Text data to fit and transform
        """
        self.fit(texts)
        self.transform_texts(texts)

    def as_retriever(self) -> "TFIDFRetriever":
        """Get a TFIDFRetriever instance using this vector store.

        Returns:
            TFIDFRetriever: Retriever instance
        """
        if not self.is_fitted:
            logger.error("Vectorizer must be fitted before creating retriever")
            raise RuntimeError("Vectorizer not fitted. Call fit() first.")
        if not self.is_transformed:
            logger.error(
                "Data must be transformed and stored before creating retriever"
            )
            raise RuntimeError(
                "Data not transformed and stored. Call transform_texts() first."
            )
        logger.info("Creating TFIDFRetriever from TFIDFVectorStore")
        return TFIDFRetriever(vector_store=self)

    def as_reranker(self) -> "TFIDFReranker":
        """Get a TFIDFReranker instance using this vector store.

        Returns:
            TFIDFReranker: Reranker instance
        """
        if not self.is_fitted:
            logger.error("Vectorizer must be fitted before creating reranker")
            raise RuntimeError("Vectorizer not fitted. Call fit() first.")

        logger.info("Creating TFIDFReranker from TFIDFVectorStore")
        return TFIDFReranker(vector_store=self)


class TFIDFRetriever(BaseRetriever):
    def __init__(self, vector_store: TFIDFVectorStore):
        """Initialize the TFIDFRetriever.

        Args:
            vector_store: An instance of TFIDFVectorStore

        Raises:
            RuntimeError: If vector_store is not fitted or transformed
        """
        super().__init__(vector_store)
        # Store as TFIDFVectorStore specifically for type safety
        self.vector_store: TFIDFVectorStore = vector_store

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Retrieve top-k similar texts for the given query.

        Args:
            query: Query text
            top_k: Number of top similar texts to retrieve
        Returns:
            list[tuple[str, float]]: List of tuples containing text and similarity score
        """
        logger.info(f"Retrieving top {top_k} results for query: {query}")

        # Transform query using the vectorizer
        query_vector = self.vector_store.transform(query)

        # Compute cosine similarity between query and all stored vectors
        similarities = cosine_similarity(
            np.expand_dims(query_vector, axis=0),
            self.vector_store.vectors,  # type: ignore
        ).flatten()

        # Get indices of top-k most similar texts
        top_k_indices = np.argsort(similarities)[::-1][:top_k]

        # Get the texts and their similarity scores
        results = [
            (self.vector_store.texts[idx], float(similarities[idx]))  # type: ignore
            for idx in top_k_indices
        ]

        logger.success(f"Retrieved {len(results)} results")

        return results


class TFIDFReranker:
    def __init__(self, vector_store: TFIDFVectorStore):
        """Initialize the TFIDFReranker.

        Args:
            vector_store: An instance of TFIDFVectorStore

        Raises:
            RuntimeError: If vector_store is not fitted
        """
        logger.debug("Initializing TFIDFReranker")

        if not vector_store.is_fitted:
            logger.error("Vector store must be fitted before creating reranker")
            raise RuntimeError("Vector store not fitted. Call fit() first.")

        self.vector_store = vector_store
        logger.info("TFIDFReranker initialized")

    def rerank(
        self,
        query: str,
        texts: list[str],
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """Rerank the given texts based on their similarity to the query.

        Args:
            query: The query string
            texts: List of texts to rerank
            top_k: Number of top relevant texts to return

        Returns:
            list[tuple[str, float]]: List of tuples containing the text and its similarity score
        """
        logger.info(f"Reranking {len(texts)} texts for query: {query}")

        # Transform query using the vectorizer
        query_vector = self.vector_store.transform(query)

        # Transform texts using the vectorizer
        texts_vectors = np.array([self.vector_store.transform(text) for text in texts])

        # Compute cosine similarity between query and all text vectors
        similarities = cosine_similarity(
            np.expand_dims(query_vector, axis=0),
            texts_vectors,
        ).flatten()

        # Get indices of top-k most similar texts
        top_k_indices = np.argsort(similarities)[::-1][:top_k]

        # Get the texts and their similarity scores
        results = [
            (texts[idx], float(similarities[idx]))  # type: ignore
            for idx in top_k_indices
        ]

        logger.success(f"Reranked and retrieved {len(results)} results")

        return results


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Example usage
    sample_texts = [
        "The cat sat on the mat.",
        "Dogs are great pets.",
        "I love programming in Python.",
        "The sun is shining today.",
        "Artificial Intelligence is the future.",
        "Coding makes me happy.",
    ]

    vector_store = TFIDFVectorStore()
    vector_store.vectorize(sample_texts)
    logger.info(vector_store.feature_names_)

    logger.info(f"Stored texts: {vector_store.texts[0]}")  # type: ignore
    logger.info(f"Stored vector for first text: {vector_store.vectors[0]}")  # type: ignore

    retriever = vector_store.as_retriever()

    query = "I enjoy coding."
    results = retriever.retrieve(query, top_k=2)

    for text, score in results:
        logger.info(f"Text: {text} | Similarity Score: {score:.4f}")
