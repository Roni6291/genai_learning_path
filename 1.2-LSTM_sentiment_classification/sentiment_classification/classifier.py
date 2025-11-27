"""Bidirectional LSTM Sentiment Classifier."""

import numpy as np
from loguru import logger
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers


class SentimentClassifier:
    """Bidirectional LSTM model for sentiment classification."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        max_length: int = 300,
        lstm_units: int = 128,
        dropout_rate: float = 0.5,
        recurrent_dropout: float = 0.2,
    ):
        """Initialize the sentiment classifier.

        Args:
            vocab_size (int): Size of vocabulary (including padding token).
            embedding_dim (int): Dimension of word embeddings.
            max_length (int): Maximum sequence length.
            lstm_units (int): Number of LSTM units. Default is 128.
            dropout_rate (float): Dropout rate for regularization. Default is 0.5.
            recurrent_dropout (float): Recurrent dropout rate. Default is 0.2.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.model = None

    def build_model(
        self,
        embedding_matrix: np.ndarray | None = None,
        freeze_embeddings: bool = True,
    ) -> 'SentimentClassifier':
        """Build the Bidirectional LSTM model.

        Args:
            embedding_matrix (np.ndarray | None): Pretrained embedding matrix.
                If None, embeddings will be learned from scratch.
            freeze_embeddings (bool): Whether to freeze embedding weights.
                Default is True (use pretrained embeddings as-is).

        Returns:
            SentimentClassifier: Self for method chaining.
        """
        logger.info('Building Bidirectional LSTM model...')

        # Input layer
        inputs = layers.Input(shape=(self.max_length,), name='input')

        # Embedding layer
        if embedding_matrix is not None:
            logger.info(f'Using pretrained embeddings (frozen={freeze_embeddings})')
            embedding = layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                weights=[embedding_matrix],
                input_length=self.max_length,
                trainable=not freeze_embeddings,
                name='embedding',
            )(inputs)
        else:
            logger.info('Learning embeddings from scratch')
            embedding = layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_length,
                name='embedding',
            )(inputs)

        # LSTM layers
        x = layers.LSTM(
            self.lstm_units,
            return_sequences=True,
            dropout=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout,
            name='lstm_1',
        )(embedding)

        x = layers.LSTM(
            self.lstm_units // 2,
            dropout=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout,
            name='lstm_2',
        )(x)

        # Dropout layer
        x = layers.Dropout(self.dropout_rate, name='dropout')(x)

        # Output layer
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

        # Create model
        self.model = keras.Model(
            inputs=inputs, outputs=outputs, name='sentiment_classifier'
        )

        logger.success(
            f'Model built successfully with {self.model.count_params():,} parameters'
        )

        return self

    def compile_model(
        self,
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        loss: str = 'binary_crossentropy',
        metrics: list[str] | None = None,
    ) -> 'SentimentClassifier':
        """Compile the model.

        Args:
            optimizer (str): Optimizer name. Default is 'adam'.
            learning_rate (float): Learning rate. Default is 0.001.
            loss (str): Loss function. Default is 'binary_crossentropy'.
            metrics (list[str] | None): List of metrics. Default is ['accuracy'].

        Returns:
            SentimentClassifier: Self for method chaining.
        """
        if self.model is None:
            raise ValueError('Model must be built before compiling.')

        if metrics is None:
            metrics = ['accuracy']

        # Create optimizer with learning rate
        if optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'adamw':
            opt = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
        else:
            opt = optimizer

        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)
        logger.success(f'Model compiled with optimizer={optimizer}, lr={learning_rate}')

        return self

    def summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            raise ValueError('Model must be built before getting summary.')
        self.model.summary()

    def get_model(self) -> keras.Model:
        """Get the Keras model.

        Returns:
            keras.Model: The compiled Keras model.
        """
        if self.model is None:
            raise ValueError('Model must be built before retrieving.')
        return self.model

    def train(
        self,
        X_train: np.ndarray,  # noqa: N803
        y_train: np.ndarray,
        X_val: np.ndarray,  # noqa: N803
        y_val: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        callbacks: list | None = None,
    ) -> keras.callbacks.History:
        """Train the model.

        Args:
            X_train (np.ndarray): Training sequences.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray): Validation sequences.
            y_val (np.ndarray): Validation labels.
            epochs (int): Number of training epochs. Default is 10.
            batch_size (int): Batch size. Default is 32.
            callbacks (list | None): List of Keras callbacks.

        Returns:
            keras.callbacks.History: Training history.
        """
        if self.model is None:
            raise ValueError('Model must be built and compiled before training.')

        logger.info(f'Starting training: {epochs} epochs, batch_size={batch_size}')
        logger.info(f'Train samples: {len(X_train)}, Validation samples: {len(X_val)}')

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        logger.success('Training completed!')
        return history

    def evaluate(
        self,
        X_test: np.ndarray,  # noqa: N803
        y_test: np.ndarray,
        batch_size: int = 32,
    ) -> tuple[float, float]:
        """Evaluate the model.

        Args:
            X_test (np.ndarray): Test sequences.
            y_test (np.ndarray): Test labels.
            batch_size (int): Batch size. Default is 32.

        Returns:
            tuple[float, float]: Test loss and accuracy.
        """
        if self.model is None:
            raise ValueError('Model must be built before evaluation.')

        logger.info(f'Evaluating model on {len(X_test)} test samples...')
        loss, accuracy = self.model.evaluate(
            X_test, y_test, batch_size=batch_size, verbose=1
        )

        logger.success(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')
        return loss, accuracy

    def predict(
        self,
        X: np.ndarray,  # noqa: N803
        batch_size: int = 32,
    ) -> np.ndarray:
        """Make predictions.

        Args:
            X (np.ndarray): Input sequences.
            batch_size (int): Batch size. Default is 32.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        if self.model is None:
            raise ValueError('Model must be built before prediction.')

        return self.model.predict(X, batch_size=batch_size, verbose=0)

    def save(self, filepath: str) -> None:
        """Save the model.

        Args:
            filepath (str): Path to save the model.
        """
        if self.model is None:
            raise ValueError('Model must be built before saving.')

        self.model.save(filepath)
        logger.success(f'Model saved to {filepath}')

    def load(self, filepath: str) -> 'SentimentClassifier':
        """Load a saved model.

        Args:
            filepath (str): Path to the saved model.

        Returns:
            SentimentClassifier: Self for method chaining.
        """
        self.model = keras.models.load_model(filepath)
        logger.success(f'Model loaded from {filepath}')
        return self
