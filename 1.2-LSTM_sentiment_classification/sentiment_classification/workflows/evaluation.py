"""Model evaluation workflow."""

from pathlib import Path

import click
import numpy as np
from loguru import logger

from sentiment_classification.evaluator import ModelEvaluator


@click.command()
@click.option(
    '--model-path',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to the trained model.',
)
@click.option(
    '--test-dir',
    type=click.Path(exists=True, path_type=Path),
    default=Path('data/test'),
    help='Directory containing test data (sequences.npy and labels.npy).',
)
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    default=Path('artifacts/evaluation'),
    help='Directory to save evaluation results.',
)
@click.option(
    '--batch-size',
    type=int,
    default=32,
    help='Batch size for predictions. Default is 32.',
)
def evaluate_model(
    model_path: Path,
    test_dir: Path,
    output_dir: Path,
    batch_size: int,
) -> None:
    """Evaluate a trained sentiment classification model.

    This command:
    1. Loads the trained model
    2. Loads test data
    3. Makes predictions
    4. Calculates evaluation metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
    5. Generates confusion matrix plot
    6. Generates ROC curve plot
    7. Saves all results to artifacts
    """
    # Load test data
    logger.info(f'Loading test data from {test_dir}')
    X_test = np.load(test_dir / 'sequences.npy')
    y_test = np.load(test_dir / 'labels.npy')
    logger.success(f'Loaded {len(X_test)} test samples')

    # Initialize evaluator and load model
    evaluator = ModelEvaluator(model_path=str(model_path))
    evaluator.load_model()

    # Make predictions
    evaluator.predict(X_test=X_test, batch_size=batch_size)

    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_true=y_test)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_path = output_dir / 'metrics.txt'
    evaluator.save_metrics(metrics=metrics, output_path=metrics_path)

    # Plot and save confusion matrix
    confusion_matrix_path = output_dir / 'confusion_matrix.png'
    evaluator.plot_confusion_matrix(output_path=confusion_matrix_path)

    # Plot and save ROC curve
    roc_curve_path = output_dir / 'roc_curve.png'
    evaluator.plot_roc_curve(output_path=roc_curve_path)

    logger.success(
        f'Evaluation complete!\n'
        f'  Metrics: {metrics_path}\n'
        f'  Confusion Matrix: {confusion_matrix_path}\n'
        f'  ROC Curve: {roc_curve_path}'
    )
