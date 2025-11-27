"""Model evaluation utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from tensorflow import keras  # type: ignore


class ModelEvaluator:
    """Evaluate sentiment classification model with metrics and visualizations."""

    CLASSIFICATION_THRESHOLD = 0.5

    def __init__(self, model_path: str):
        """Initialize the evaluator with a trained model.

        Args:
            model_path (str): Path to the saved Keras model.
        """
        self.model_path = model_path
        self.model = None
        self.y_true = None
        self.y_pred = None
        self.y_pred_proba = None

    def load_model(self) -> 'ModelEvaluator':
        """Load the trained model.

        Returns:
            ModelEvaluator: Self for method chaining.
        """
        logger.info(f'Loading model from {self.model_path}')
        self.model = keras.models.load_model(self.model_path)
        logger.success('Model loaded successfully')
        return self

    def predict(
        self,
        X_test: np.ndarray,  # noqa: N803
        batch_size: int = 32,
    ) -> 'ModelEvaluator':
        """Make predictions on test data.

        Args:
            X_test (np.ndarray): Test sequences.
            batch_size (int): Batch size for prediction. Default is 32.

        Returns:
            ModelEvaluator: Self for method chaining.
        """
        if self.model is None:
            raise ValueError('Model must be loaded before making predictions.')

        logger.info(f'Making predictions on {len(X_test)} test samples...')
        self.y_pred_proba = self.model.predict(X_test, batch_size=batch_size, verbose=0)
        self.y_pred = (
            (self.y_pred_proba > self.CLASSIFICATION_THRESHOLD).astype(int).flatten()
        )
        logger.success('Predictions completed')
        return self

    def calculate_metrics(self, y_true: np.ndarray) -> dict[str, float]:
        """Calculate evaluation metrics.

        Args:
            y_true (np.ndarray): True labels.

        Returns:
            dict[str, float]: Dictionary containing all metrics.
        """
        if self.y_pred is None or self.y_pred_proba is None:
            raise ValueError('Predictions must be made before calculating metrics.')

        self.y_true = y_true

        # Calculate metrics
        accuracy = accuracy_score(y_true, self.y_pred)
        precision = precision_score(y_true, self.y_pred)
        recall = recall_score(y_true, self.y_pred)
        f1 = f1_score(y_true, self.y_pred)

        # Calculate ROC-AUC
        fpr, tpr, _ = roc_curve(y_true, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
        }

        # Log metrics
        logger.success('Evaluation Metrics:')
        logger.info(f'  Accuracy:  {accuracy:.4f}')
        logger.info(f'  Precision: {precision:.4f}')
        logger.info(f'  Recall:    {recall:.4f}')
        logger.info(f'  F1-Score:  {f1:.4f}')
        logger.info(f'  ROC-AUC:   {roc_auc:.4f}')

        return metrics

    def plot_confusion_matrix(self, output_path: Path) -> 'ModelEvaluator':
        """Plot and save confusion matrix.

        Args:
            output_path (Path): Path to save the confusion matrix plot.

        Returns:
            ModelEvaluator: Self for method chaining.
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError(
                'Metrics must be calculated before plotting confusion matrix.'
            )

        logger.info('Generating confusion matrix...')

        # Calculate confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)

        # Create figure
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            cbar_kws={'label': 'Count'},
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        # Save plot
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.success(f'Confusion matrix saved to {output_path}')
        return self

    def plot_roc_curve(self, output_path: Path) -> 'ModelEvaluator':
        """Plot and save ROC curve.

        Args:
            output_path (Path): Path to save the ROC curve plot.

        Returns:
            ModelEvaluator: Self for method chaining.
        """
        if self.y_true is None or self.y_pred_proba is None:
            raise ValueError('Metrics must be calculated before plotting ROC curve.')

        logger.info('Generating ROC curve...')

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Create figure
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            color='darkorange',
            lw=2,
            label=f'ROC curve (AUC = {roc_auc:.4f})',
        )
        plt.plot(
            [0, 1],
            [0, 1],
            color='navy',
            lw=2,
            linestyle='--',
            label='Random Classifier',
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(
            'Receiver Operating Characteristic (ROC) Curve',
            fontsize=16,
            fontweight='bold',
        )
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # Save plot
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.success(f'ROC curve saved to {output_path}')
        return self

    def save_metrics(
        self, metrics: dict[str, float], output_path: Path
    ) -> 'ModelEvaluator':
        """Save metrics to a text file.

        Args:
            metrics (dict[str, float]): Dictionary of metrics.
            output_path (Path): Path to save the metrics file.

        Returns:
            ModelEvaluator: Self for method chaining.
        """
        logger.info(f'Saving metrics to {output_path}')

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('Model Evaluation Metrics\n')
            f.write('=' * 40 + '\n\n')
            for key, value in metrics.items():
                f.write(f'{key.replace("_", " ").title()}: {value:.6f}\n')

        logger.success(f'Metrics saved to {output_path}')
        return self
