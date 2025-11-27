import numpy as np
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from tensorflow.keras.preprocessing.sequence import pad_sequences  # noqa: E402
from tensorflow.keras.preprocessing.text import Tokenizer  # noqa: E402


class ReviewTokenizer:
    """A class for tokenizing and padding text reviews using state in/state out approach.

    This class maintains internal state for texts, sequences, and padded sequences,
    allowing method chaining for a fluent API.
    """

    def __init__(self, num_words: int = 10000, oov_token: str = '<OOV>'):
        """Initialize the ReviewTokenizer.

        Args:
            num_words (int): Maximum number of words to keep in vocabulary.
            oov_token (str): Token to use for out-of-vocabulary words.
        """
        self.tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        self.num_words = num_words
        self.oov_token = oov_token
        self.texts: list[str] = []
        self.sequences: list[list[int]] = []
        self.padded_sequences: np.ndarray | None = None
        self.is_fitted = False

    def set_texts(self, texts: list[str]) -> 'ReviewTokenizer':
        """Set the texts to be tokenized.

        Args:
            texts (list[str]): List of text reviews.
        Returns:
            ReviewTokenizer: Self for method chaining.
        """
        self.texts = texts
        return self

    def fit(self, texts: list[str] | None = None) -> 'ReviewTokenizer':
        """Fit the tokenizer on texts to build vocabulary.

        Args:
            texts (list[str] | None): List of texts to fit on. If None, uses internal state.
        Returns:
            ReviewTokenizer: Self for method chaining.
        """
        if texts is not None:
            self.texts = texts
        self.tokenizer.fit_on_texts(self.texts)
        self.is_fitted = True
        return self

    def texts_to_sequences(self, texts: list[str] | None = None) -> 'ReviewTokenizer':
        """Convert texts to sequences of integer indices.

        Args:
            texts (list[str] | None): List of texts to convert. If None, uses internal state.
        Returns:
            ReviewTokenizer: Self for method chaining.
        """
        if not self.is_fitted:
            raise ValueError(
                'Tokenizer must be fitted before converting texts to sequences.'
            )

        if texts is not None:
            self.texts = texts
        self.sequences = self.tokenizer.texts_to_sequences(self.texts)
        return self

    def pad_sequences(
        self,
        sequences: list[list[int]] | None = None,
        maxlen: int = 250,
        padding: str = 'post',
        truncating: str = 'post',
    ) -> 'ReviewTokenizer':
        """Pad or truncate sequences to a fixed length.

        Args:
            sequences (list[list[int]] | None): Sequences to pad. If None, uses internal state.
            maxlen (int): Maximum length of sequences. Default is 250.
            padding (str): 'pre' or 'post' padding. Default is 'post'.
            truncating (str): 'pre' or 'post' truncating. Default is 'post'.
        Returns:
            ReviewTokenizer: Self for method chaining.
        """
        if sequences is not None:
            self.sequences = sequences
        self.padded_sequences = pad_sequences(
            self.sequences, maxlen=maxlen, padding=padding, truncating=truncating
        )
        return self

    def get_sequences(self) -> list[list[int]]:
        """Get the current sequences.

        Returns:
            list[list[int]]: The sequences of integer indices.
        """
        return self.sequences

    def get_padded_sequences(self) -> np.ndarray:
        """Get the padded sequences.

        Returns:
            np.ndarray: The padded sequences array.
        """
        if self.padded_sequences is None:
            raise ValueError('Sequences must be padded before retrieving them.')
        return self.padded_sequences

    def get_vocabulary(self) -> dict[str, int]:
        """Get the word index vocabulary.

        Returns:
            dict[str, int]: Dictionary mapping words to their integer indices.
        """
        if not self.is_fitted:
            raise ValueError('Tokenizer must be fitted before accessing vocabulary.')
        return self.tokenizer.word_index

    def get_vocabulary_size(self) -> int:
        """Get the size of the vocabulary actually being used (limited by num_words).

        Returns:
            int: Number of unique words in vocabulary (capped at num_words + 1 for padding).
        """
        if not self.is_fitted:
            raise ValueError(
                'Tokenizer must be fitted before accessing vocabulary size.'
            )
        # Return the effective vocabulary size (limited by num_words)
        full_vocab_size = len(self.tokenizer.word_index)
        if self.num_words is not None and full_vocab_size > self.num_words:
            return self.num_words + 1  # +1 for padding token (index 0)
        return full_vocab_size + 1  # +1 for padding token
