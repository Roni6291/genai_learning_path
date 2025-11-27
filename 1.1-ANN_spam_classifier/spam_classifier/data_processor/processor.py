"""Text vectorization module for spam classifier.

Provides TextProcessor class for TF-IDF vectorization of text data.
"""

import pickle
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Constants
EXPECTED_COLUMNS = 2


class TextProcessor:
    """A class for vectorizing text data using TF-IDF.

    This class handles loading tab-delimited data files and transforming
    text into TF-IDF feature vectors suitable for machine learning.
    """

    def __init__(
        self,
        max_features: int = 5000,
        min_df: int = 2,
        max_df: float = 1.0,
        ngram_range: tuple[int, int] = (1, 2),
    ):
        """Initialize the TextProcessor.

        Args:
            max_features: Maximum number of features (words) to extract.
                Default is 5000.
            min_df: Minimum document frequency for a word to be included.
                Words appearing in fewer documents will be ignored. Default is 2.
            max_df: Maximum document frequency (as proportion). Words appearing
                in more than this proportion of documents will be ignored.
            ngram_range: Range of n-grams to extract. (1, 1) for unigrams,
                (1, 2) for unigrams and bigrams (default), etc.
        """
        logger.debug('Initializing TextProcessor')
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
        )

        self.is_fitted = False
        self.feature_names_ = None
        self.label_encoder = LabelEncoder()
        self.is_label_encoder_fitted = False
        logger.info(
            f'TextProcessor initialized with max_features={max_features}, '
            f'min_df={min_df}, max_df={max_df}, ngram_range={ngram_range}'
        )

    def _validate_line(self, parts: list[str], line_num: int) -> None:
        """Validate a line has the correct number of columns.

        Args:
            parts: Split line parts
            line_num: Line number for error reporting

        Raises:
            ValueError: If line doesn't have expected number of columns
        """
        if len(parts) != EXPECTED_COLUMNS:
            msg = (
                f'Line {line_num}: Expected {EXPECTED_COLUMNS} columns '
                f'(label, text), got {len(parts)}'
            )
            raise ValueError(msg)

    def _validate_data(self, labels: list[str]) -> None:
        """Validate that data was loaded successfully.

        Args:
            labels: List of loaded labels

        Raises:
            ValueError: If no valid data was found
        """
        if not labels:
            msg = 'No valid data found in file'
            raise ValueError(msg)

    def load_data(self, file_path: str | Path) -> tuple[list[str], list[str]]:
        """Load tab-delimited data from a file.

        Expected format: label<tab>text
        First column is the label, second column is the text.

        Args:
            file_path: Path to the tab-delimited data file

        Returns:
            tuple: (labels, texts) as lists of strings

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is incorrect
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f'File not found: {file_path}')
            raise FileNotFoundError(f'File not found: {file_path}')

        logger.info(f'Loading data from {file_path}')

        labels = []
        texts = []

        try:
            with open(file_path, encoding='utf-8') as f:
                for line_num, raw_line in enumerate(f, 1):
                    line = raw_line.rstrip('\n\r')
                    if not line:  # Skip empty lines
                        continue

                    parts = line.split('\t')
                    self._validate_line(parts, line_num)

                    label, text = parts
                    if not label or not text:
                        logger.warning(
                            f'Line {line_num}: Skipping line with missing values'
                        )
                        continue

                    labels.append(label)
                    texts.append(text)
        except Exception as e:
            logger.error(f'Error loading data: {e}')
            raise
        else:
            self._validate_data(labels)
            logger.success(f'Loaded {len(labels)} samples from {file_path}')
            return labels, texts

    def fit(self, texts: list[str]) -> 'TextProcessor':
        """Fit the TF-IDF vectorizer on the training data.

        Args:
            texts: Text data to fit the vectorizer on

        Returns:
            self: Returns the instance for method chaining
        """
        logger.info(f'Fitting vectorizer on {len(texts)} samples')

        self.vectorizer.fit(texts)
        self.is_fitted = True
        self.feature_names_ = self.vectorizer.get_feature_names_out()

        logger.info(f'Vectorizer fitted. Generated {len(self.feature_names_)} features')
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform text data into TF-IDF features.

        Args:
            texts: Text data to transform

        Returns:
            np.ndarray: TF-IDF feature matrix

        Raises:
            RuntimeError: If vectorizer hasn't been fitted yet
        """
        if not self.is_fitted:
            logger.error('Vectorizer must be fitted before transform')
            raise RuntimeError(
                'Vectorizer not fitted. Call fit() or fit_transform() first.'
            )

        logger.info(f'Transforming {len(texts)} samples')
        tfidf_matrix = self.vectorizer.transform(texts)

        logger.info(f'Transformed to shape {tfidf_matrix.shape} (samples x features)')
        return tfidf_matrix.toarray()  # type: ignore[attr-defined]

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Fit the vectorizer and transform the data in one step.

        Args:
            texts: Text data to fit and transform

        Returns:
            np.ndarray: TF-IDF feature matrix
        """
        logger.info(f'Fitting and transforming {len(texts)} samples')

        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        self.feature_names_ = self.vectorizer.get_feature_names_out()

        logger.info(
            f'Fit-transform complete. Shape: {tfidf_matrix.shape}, '
            f'Features: {len(self.feature_names_)}'
        )
        return tfidf_matrix.toarray()  # type: ignore[attr-defined]

    def get_feature_names(self) -> np.ndarray | None:
        """Get the feature names (vocabulary) learned by the vectorizer.

        Returns:
            np.ndarray: Array of feature names, or None if not fitted

        Raises:
            RuntimeError: If vectorizer hasn't been fitted yet
        """
        if not self.is_fitted:
            logger.error('Vectorizer must be fitted to get feature names')
            raise RuntimeError(
                'Vectorizer not fitted. Call fit() or fit_transform() first.'
            )

        return self.feature_names_

    def save_vectorizer(self, file_path: str | Path) -> None:
        """Save the fitted vectorizer to disk using pickle.

        Args:
            file_path: Path where to save the vectorizer

        Raises:
            RuntimeError: If vectorizer hasn't been fitted yet
        """
        if not self.is_fitted:
            logger.error('Cannot save unfitted vectorizer')
            raise RuntimeError(
                'Vectorizer not fitted. Call fit() or fit_transform() first.'
            )

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f'Saving vectorizer to {file_path}')
        with open(file_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

        logger.success('Vectorizer saved successfully')

    def load_vectorizer(self, file_path: str | Path) -> 'TextProcessor':
        """Load a fitted vectorizer from disk.

        Args:
            file_path: Path to the saved vectorizer

        Returns:
            self: Returns the instance for method chaining

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f'Vectorizer file not found: {file_path}')
            raise FileNotFoundError(f'Vectorizer file not found: {file_path}')

        logger.debug(f'Loading vectorizer from {file_path}')
        with open(file_path, 'rb') as f:
            self.vectorizer = pickle.load(f)

        self.is_fitted = True
        self.feature_names_ = self.vectorizer.get_feature_names_out()

        logger.success(
            f'Vectorizer loaded successfully with {len(self.feature_names_)} features'
        )
        return self

    def get_vocabulary_size(self) -> int:
        """Get the size of the vocabulary.

        Returns:
            int: Number of features in the vocabulary

        Raises:
            RuntimeError: If vectorizer hasn't been fitted yet
        """
        if not self.is_fitted:
            logger.error('Vectorizer must be fitted to get vocabulary size')
            raise RuntimeError(
                'Vectorizer not fitted. Call fit() or fit_transform() first.'
            )

        if self.feature_names_ is None:
            return 0
        return len(self.feature_names_)

    def encode_labels(self, labels: list[str]) -> np.ndarray:
        """Encode text labels to numeric values.

        Args:
            labels: List of text labels to encode

        Returns:
            np.ndarray: Encoded numeric labels
        """
        logger.info(f'Encoding {len(labels)} labels')
        encoded = self.label_encoder.fit_transform(labels)
        self.is_label_encoder_fitted = True

        # Log the label mapping
        classes = self.label_encoder.classes_
        encoded_values = list(range(len(classes)))
        label_mapping = dict(zip(classes, encoded_values, strict=True))
        logger.info(f'Label mapping: {label_mapping}')

        return np.array(encoded)

    def decode_labels(self, encoded_labels: np.ndarray) -> np.ndarray:
        """Decode numeric labels back to text.

        Args:
            encoded_labels: Numeric encoded labels

        Returns:
            np.ndarray: Decoded text labels

        Raises:
            RuntimeError: If label encoder hasn't been fitted yet
        """
        if not self.is_label_encoder_fitted:
            logger.error('Label encoder must be fitted before decode')
            raise RuntimeError('Label encoder not fitted. Call encode_labels() first.')

        return self.label_encoder.inverse_transform(encoded_labels)

    def get_label_classes(self) -> np.ndarray | None:
        """Get the label classes learned by the encoder.

        Returns:
            np.ndarray: Array of label classes, or None if not fitted

        Raises:
            RuntimeError: If label encoder hasn't been fitted yet
        """
        if not self.is_label_encoder_fitted:
            logger.error('Label encoder must be fitted to get classes')
            raise RuntimeError('Label encoder not fitted. Call encode_labels() first.')

        return self.label_encoder.classes_

    def save_label_encoder(self, file_path: str | Path) -> None:
        """Save the fitted label encoder to disk using pickle.

        Args:
            file_path: Path where to save the label encoder

        Raises:
            RuntimeError: If label encoder hasn't been fitted yet
        """
        if not self.is_label_encoder_fitted:
            logger.error('Cannot save unfitted label encoder')
            raise RuntimeError('Label encoder not fitted. Call encode_labels() first.')

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f'Saving label encoder to {file_path}')
        with open(file_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)

        logger.success('Label encoder saved successfully')

    def load_label_encoder(self, file_path: str | Path) -> 'TextProcessor':
        """Load a fitted label encoder from disk.

        Args:
            file_path: Path to the saved label encoder

        Returns:
            self: Returns the instance for method chaining

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f'Label encoder file not found: {file_path}')
            raise FileNotFoundError(f'Label encoder file not found: {file_path}')

        logger.debug(f'Loading label encoder from {file_path}')
        with open(file_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        self.is_label_encoder_fitted = True

        logger.success(
            f'Label encoder loaded successfully with classes: {self.label_encoder.classes_}'
        )
        return self


def preprocess_and_save(
    input_file: str | Path,
    output_dir: str | Path = 'data/preprocess',
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorize data and save to preprocess directory.

    Args:
        input_file: Path to input data file (tab-delimited)
        output_dir: Directory to save preprocessed files (default: 'data/preprocess')
        max_features: Maximum number of features to extract
        ngram_range: Range of n-grams to extract (default: (1, 2) for unigrams + bigrams)
        min_df: Minimum document frequency (default: 2)

    Returns:
        tuple: (X, y) - vectorized features and labels
    """
    vectorizer = TextProcessor(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )

    # Load and vectorize data
    labels, texts = vectorizer.load_data(input_file)
    X = vectorizer.fit_transform(texts)
    y = vectorizer.encode_labels(labels)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save vectorized features
    features_file = output_path / 'features.npy'
    np.save(features_file, X)
    logger.info(f'Saved features to {features_file}')

    # Save encoded labels
    labels_file = output_path / 'labels.npy'
    np.save(labels_file, y)
    logger.info(f'Saved encoded labels to {labels_file}')

    # Save vectorizer and label encoder to artifacts directory
    artifacts_dir = Path('artifacts')
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    vectorizer_file = artifacts_dir / 'vectorizer.pkl'
    vectorizer.save_vectorizer(vectorizer_file)

    label_encoder_file = artifacts_dir / 'label_encoder.pkl'
    vectorizer.save_label_encoder(label_encoder_file)

    logger.success(f'Preprocessing complete. Saved {len(y)} samples to {output_path}')
    logger.success(f'Saved vectorizer and label encoder to {artifacts_dir}')

    return X, y
