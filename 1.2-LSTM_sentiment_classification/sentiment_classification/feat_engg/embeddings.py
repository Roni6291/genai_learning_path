"""Embedding utilities for loading and building embedding matrices."""

import numpy as np
from gensim.models import KeyedVectors
from loguru import logger


class EmbeddingMatrixBuilder:
    """Build embedding matrix from pretrained Word2Vec embeddings."""

    def __init__(
        self,
        word2vec_path: str,
        embedding_dim: int = 300,
        limit: int | None = None,
    ):
        """Initialize the embedding matrix builder.

        Args:
            word2vec_path (str): Path to pretrained Word2Vec binary file.
            embedding_dim (int): Dimension of word embeddings. Default is 300.
            limit (int | None): Limit number of word vectors to load (for faster loading during testing).
        """
        self.word2vec_path = word2vec_path
        self.embedding_dim = embedding_dim
        self.limit = limit
        self.word2vec_model = None

    def load_word2vec(self) -> 'EmbeddingMatrixBuilder':
        """Load pretrained Word2Vec embeddings.

        Returns:
            EmbeddingMatrixBuilder: Self for method chaining.
        """
        logger.info(f'Loading Word2Vec embeddings from {self.word2vec_path}')
        if self.limit:
            logger.info(f'Limiting to first {self.limit} vectors for faster loading')

        self.word2vec_model = KeyedVectors.load_word2vec_format(
            self.word2vec_path, binary=True, limit=self.limit
        )
        logger.success(
            f'Loaded {len(self.word2vec_model)} word vectors '
            f'with dimension {self.word2vec_model.vector_size}'
        )
        return self

    def build_embedding_matrix(
        self, word_index: dict[str, int], vocab_size: int
    ) -> np.ndarray:
        """Build embedding matrix from vocabulary and Word2Vec model.

        Args:
            word_index (dict[str, int]): Dictionary mapping words to indices.
            vocab_size (int): Size of vocabulary (including padding token).

        Returns:
            np.ndarray: Embedding matrix of shape (vocab_size, embedding_dim).
        """
        if self.word2vec_model is None:
            raise ValueError('Word2Vec model must be loaded before building matrix.')

        logger.info(f'Building embedding matrix for vocabulary size {vocab_size}')

        # Initialize embedding matrix with random values
        embedding_matrix = np.random.uniform(
            low=-0.05, high=0.05, size=(vocab_size, self.embedding_dim)
        )

        # Set padding token (index 0) to zeros
        embedding_matrix[0] = np.zeros(self.embedding_dim)

        # Fill in pretrained embeddings
        found_count = 0
        oov_count = 0

        for word, idx in word_index.items():
            if idx >= vocab_size:
                continue

            if word in self.word2vec_model:
                embedding_matrix[idx] = self.word2vec_model[word]
                found_count += 1
            else:
                oov_count += 1

        logger.success(
            f'Embedding matrix built: {found_count} words found in Word2Vec, '
            f'{oov_count} OOV words initialized randomly'
        )
        logger.info(
            f'Coverage: {found_count / (found_count + oov_count) * 100:.2f}% '
            f'of vocabulary found in pretrained embeddings'
        )

        return embedding_matrix
