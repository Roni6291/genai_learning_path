"""SMS Spam Classifier using TensorFlow Keras."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import find_dotenv, load_dotenv
from loguru import logger

load_dotenv(find_dotenv())

from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402


class SMSSpamClassifier:
    """Binary classifier for SMS spam detection using neural networks.

    Architecture:
        - Input layer: size equal to TF-IDF features
        - Hidden layers: 1-2 layers with 64-128 neurons each, ReLU activation
        - Output layer: 1 neuron with Sigmoid activation for binary classification

    Args:
        input_dim: Number of input features (TF-IDF vocabulary size)
        hidden_layers: List of hidden layer sizes (default: [128, 64])
        dropout_rate: Dropout rate for regularization (default: 0.3)
        learning_rate: Learning rate for Adam optimizer (default: 0.001)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int] | None = None,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
    ):
        """Initialize the SMS spam classifier.

        Args:
            input_dim: Number of input features from TF-IDF vectorization
            hidden_layers: List of neurons for each hidden layer
            dropout_rate: Dropout rate for regularization (0.0 to 1.0)
            learning_rate: Learning rate for optimizer
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers or [128, 64]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model: keras.Model | None = None

        self._build_model()

    def _build_model(self) -> None:
        """Build the neural network architecture."""
        model = keras.Sequential(name='sms_spam_classifier')

        # Input layer
        model.add(layers.Input(shape=(self.input_dim,), name='input'))

        # Hidden layers with ReLU activation and dropout
        for i, units in enumerate(self.hidden_layers, 1):
            model.add(
                layers.Dense(
                    units,
                    activation='relu',
                    name=f'hidden_{i}',
                )
            )
            if self.dropout_rate > 0:
                model.add(layers.Dropout(self.dropout_rate, name=f'dropout_{i}'))

        # Output layer with sigmoid activation for binary classification
        model.add(layers.Dense(1, activation='sigmoid', name='output'))

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()],
        )

        self.model = model

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 10,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> keras.callbacks.History:
        """Train the model.

        Args:
            x_train: Training features
            y_train: Training labels
            x_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity mode (0, 1, or 2)

        Returns:
            Training history object
        """
        if self.model is None:
            raise ValueError('Model not built. Call _build_model() first.')

        validation_data = None
        if x_val is not None and y_val is not None:
            validation_data = (x_val, y_val)

        return self.model.fit(
            x_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Make predictions on input data.

        Args:
            x: Input features
            threshold: Classification threshold (default: 0.5)

        Returns:
            Binary predictions (0 or 1)
        """
        if self.model is None:
            raise ValueError('Model not built or trained.')

        probabilities = self.model.predict(x)
        predictions = (probabilities >= threshold).astype(int)
        return predictions.flatten()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Get prediction probabilities.

        Args:
            x: Input features

        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError('Model not built or trained.')

        return self.model.predict(x).flatten()

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model on test data.

        Args:
            x_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError('Model not built or trained.')

        results = self.model.evaluate(x_test, y_test, verbose=0)
        # Ensure results is a list for proper zipping
        if not isinstance(results, list):
            results = [results]
        return dict(zip(self.model.metrics_names, results, strict=False))

    def summary(self) -> None:
        """Print model architecture summary."""
        if self.model is None:
            raise ValueError('Model not built.')

        self.model.summary()

    def save(self, filepath: str) -> None:
        """Save model to file.

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError('Model not built or trained.')

        self.model.save(filepath)

    def load(self, filepath: str) -> None:
        """Load model from file.

        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)

    def plot_training_history(
        self,
        history: keras.callbacks.History,
        save_path: str | None = None,
        show_plot: bool = False,
    ) -> None:
        """Plot training vs validation loss and accuracy curves.

        Args:
            history: Training history object returned from train()
            save_path: Optional path to save the plot image
            show_plot: Whether to display the plot (default: False)
        """
        # Create figure with 2 subplots
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Get number of epochs
        epochs_range = range(1, len(history.history['loss']) + 1)

        # Plot loss
        ax1.plot(epochs_range, history.history['loss'], 'b-', label='Training Loss')
        if 'val_loss' in history.history:
            ax1.plot(
                epochs_range, history.history['val_loss'], 'r-', label='Validation Loss'
            )
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Plot accuracy
        ax2.plot(
            epochs_range, history.history['accuracy'], 'b-', label='Training Accuracy'
        )
        if 'val_accuracy' in history.history:
            ax2.plot(
                epochs_range,
                history.history['val_accuracy'],
                'r-',
                label='Validation Accuracy',
            )
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)

        # Adjust layout
        plt.tight_layout()

        # Save plot if path provided
        if save_path:
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f'Training history plot saved to: {save_path}')
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
