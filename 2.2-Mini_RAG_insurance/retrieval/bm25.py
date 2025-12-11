import sys
from collections import Counter
from math import log

import nltk
import numpy as np
from loguru import logger

from .base import BaseRetriever, BaseVectorStore

nltk.download("punkt")


class BM25PlusVectorStore(BaseVectorStore):
    """BM25Plus vector store implementation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 1.0):
        """Initialize BM25Plus vector store.

        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
            delta: BM25+ delta parameter (default: 1.0)
        """
        logger.debug("Initializing BM25PlusVectorStore")
        self.k1 = k1
        self.b = b
        self.delta = delta

        self._texts = None
        self._vectors = None
        self._tokenized_corpus = None
        self._doc_lengths = None
        self._avg_doc_length = None
        self._idf = None
        self._vocabulary = None

        logger.debug(
            f"BM25PlusVectorStore initialized with k1={k1}, b={b}, delta={delta}"
        )

    @property
    def texts(self) -> list[str] | None:
        """Get the stored texts.

        Returns:
            list[str] | None: Stored texts or None if not set
        """
        return self._texts

    @property
    def vectors(self) -> list[np.ndarray] | None:
        """Get the stored vectors.

        Returns:
            list[np.ndarray] | None: Stored vectors or None if not set
        """
        return self._vectors

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            list[str]: List of tokens
        """
        # Simple tokenization: lowercase and split on whitespace
        return nltk.word_tokenize(text.lower())

    def _compute_idf(self) -> None:
        """Compute IDF (Inverse Document Frequency) for all terms in the corpus."""
        if self._tokenized_corpus is None:
            raise RuntimeError("Corpus not tokenized. Call vectorize() first.")

        num_docs = len(self._tokenized_corpus)
        # Count document frequency for each term
        df = Counter()
        for doc in self._tokenized_corpus:
            unique_terms = set(doc)
            df.update(unique_terms)

        # Build vocabulary and compute IDF
        self._vocabulary = sorted(df.keys())
        self._idf = {}

        for term in self._vocabulary:
            # BM25 IDF formula: log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
            self._idf[term] = log((num_docs - df[term] + 0.5) / (df[term] + 0.5) + 1.0)

        logger.debug(f"Computed IDF for {len(self._vocabulary)} terms")

    def _compute_bm25plus_score(
        self,
        query_tokens: list[str],
        doc_tokens: list[str],
    ) -> float:
        """Compute BM25+ score for a document given a query.

        Args:
            query_tokens: Tokenized query
            doc_tokens: Tokenized document

        Returns:
            float: BM25+ score
        """
        score = 0.0
        doc_term_freq = Counter(doc_tokens)
        doc_length = len(doc_tokens)

        if self._idf is None:
            raise RuntimeError("IDF of corpus not computed. Call vectorize() first.")

        if self._avg_doc_length is None:
            raise RuntimeError(
                "Average document length not computed. Call vectorize() first."
            )

        for term in query_tokens:
            if term not in self._idf:
                continue

            tf = doc_term_freq.get(term, 0)
            idf = self._idf[term]

            # BM25+ formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (float(doc_length) / self._avg_doc_length)
            )

            score += idf * (numerator / denominator + self.delta)

        return score

    def vectorize(self, texts: list[str]) -> None:
        """Tokenize and compute BM25+ vectors for texts.

        Args:
            texts: Text data to vectorize
        """
        logger.info(f"Vectorizing {len(texts)} texts with BM25+")

        self._texts = texts
        self._tokenized_corpus = [self._tokenize(text) for text in texts]
        self._doc_lengths = [len(doc) for doc in self._tokenized_corpus]
        self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths)

        # Compute IDF
        self._compute_idf()

        # For BM25+, we store the tokenized documents as "vectors"
        # This is a simplification - in practice, we compute scores on-the-fly
        self._vectors = [np.array(doc) for doc in self._tokenized_corpus]

        logger.success(f"Vectorized {len(texts)} texts with BM25+")

    def get_scores(self, query: str) -> np.ndarray:
        """Get BM25+ scores for all documents given a query.

        Args:
            query: Query text

        Returns:
            np.ndarray: BM25+ scores for all documents
        """
        if self._tokenized_corpus is None:
            raise RuntimeError("Corpus not vectorized. Call vectorize() first.")

        query_tokens = self._tokenize(query)
        scores = []

        for doc_tokens in self._tokenized_corpus:
            score = self._compute_bm25plus_score(query_tokens, doc_tokens)
            scores.append(score)

        return np.array(scores)

    def as_retriever(self) -> "BM25PlusRetriever":
        """Get a BM25PlusRetriever instance using this vector store.

        Returns:
            BM25PlusRetriever: Retriever instance
        """
        if self._texts is None or self._vectors is None:
            logger.error("Vector store must be vectorized before creating retriever")
            raise RuntimeError("Vector store not vectorized. Call vectorize() first.")

        logger.info("Creating BM25PlusRetriever from BM25PlusVectorStore")
        return BM25PlusRetriever(vector_store=self)


class BM25PlusRetriever(BaseRetriever):
    """BM25Plus retriever implementation."""

    def __init__(self, vector_store: BM25PlusVectorStore):
        """Initialize the BM25PlusRetriever.

        Args:
            vector_store: An instance of BM25PlusVectorStore

        Raises:
            RuntimeError: If vector_store is not vectorized
        """
        super().__init__(vector_store)
        # Store as BM25PlusVectorStore specifically for type safety
        self.vector_store: BM25PlusVectorStore = vector_store

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Retrieve top-k documents using BM25+ scoring.

        Args:
            query: Query text
            top_k: Number of top results to retrieve

        Returns:
            list[tuple[str, float]]: List of (text, score) tuples
        """
        logger.info(f"Retrieving top {top_k} results for query: {query}")

        # Get BM25+ scores for all documents
        scores = self.vector_store.get_scores(query)

        # Get indices of top-k documents
        top_k_indices = np.argsort(scores)[::-1][:top_k]

        # Get the texts and their scores
        results = [
            (self.vector_store.texts[idx], float(scores[idx]))  # type: ignore
            for idx in top_k_indices
        ]

        logger.success(f"Retrieved {len(results)} results")

        return results


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Example usage
    sample_texts = [
        "The cat sat on the mat.",
        "Dogs are great pets.",
        "Programming in Python is fun.",
        "The sun is shining today.",
        "Artificial Intelligence is the future.",
        "Coding makes me happy.",
    ]

    vector_store = BM25PlusVectorStore()
    vector_store.vectorize(sample_texts)

    logger.info(f"Stored texts: {vector_store.texts[0]}")  # type: ignore
    logger.info(f"Vocabulary size: {len(vector_store._vocabulary)}")  # type: ignore

    retriever = vector_store.as_retriever()

    query = "Coding and having pets is fun."
    results = retriever.retrieve(query, top_k=3)

    for text, score in results:
        logger.info(f"Text: {text} | BM25+ Score: {score:.4f}")
