"""Evaluation pipeline for SMS spam classifier."""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from spam_classifier.classifier import SMSSpamClassifier


@click.command()
@click.option(
    '--model-path',
    '-m',
    type=click.Path(exists=True),
    required=True,
    help='Path to trained model file (.keras)',
)
@click.option(
    '--test-dir',
    '-t',
    type=click.Path(exists=True),
    default='data/test',
    help='Directory containing test data (default: data/test)',
)
@click.option(
    '--threshold',
    type=float,
    default=0.5,
    help='Classification threshold (default: 0.5)',
)
@click.option(
    '--output-dir',
    '-o',
    type=click.Path(),
    default='artifacts',
    help='Directory to save evaluation plots (default: artifacts)',
)
def sms_spam_classifier_evaluator(
    model_path: str,
    test_dir: str,
    threshold: float,
    output_dir: str,
) -> None:
    """Evaluate trained SMS spam classifier on test data.

    Loads a trained model and test data, evaluates performance,
    and displays detailed metrics including accuracy, precision, and recall.
    """
    logger.info('Starting SMS spam classifier evaluation')

    # Load test data
    test_path = Path(test_dir)
    logger.debug(f'Loading test data from {test_path}')

    x_test = np.load(test_path / 'X_test.npy')
    y_test = np.load(test_path / 'y_test.npy')

    logger.info(f'Test data: X={x_test.shape}, y={y_test.shape}')

    # Load trained model
    logger.debug(f'Loading trained model from {model_path}')
    input_dim = x_test.shape[1]

    classifier = SMSSpamClassifier(input_dim=input_dim)
    classifier.load(model_path)

    logger.success('Model loaded successfully')

    # Evaluate on test set
    logger.info(f'Evaluating model on test set (threshold={threshold})')
    metrics = classifier.evaluate(x_test=x_test, y_test=y_test)

    # Display evaluation results
    logger.success('=' * 60)
    logger.success('Test Set Evaluation Results:')
    logger.success('=' * 60)

    for metric_name, value in metrics.items():
        logger.info(f'{metric_name:20s}: {value:.4f}')

    logger.success('=' * 60)

    # Get predictions for additional analysis
    logger.info('Generating predictions for detailed analysis')
    predictions = classifier.predict(x_test, threshold=threshold)

    # Calculate additional metrics using sklearn
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    logger.info('\nDetailed Classification Metrics:')
    logger.info(f'  Accuracy:  {accuracy:.4f}')
    logger.info(f'  Precision: {precision:.4f}')
    logger.info(f'  Recall:    {recall:.4f}')
    logger.info(f'  F1-Score:  {f1:.4f}')

    # Calculate confusion matrix values
    true_positives = np.sum((predictions == 1) & (y_test == 1))
    true_negatives = np.sum((predictions == 0) & (y_test == 0))
    false_positives = np.sum((predictions == 1) & (y_test == 0))
    false_negatives = np.sum((predictions == 0) & (y_test == 1))

    logger.info('\nConfusion Matrix:')
    logger.info(f'  True Positives:  {true_positives:5d}')
    logger.info(f'  True Negatives:  {true_negatives:5d}')
    logger.info(f'  False Positives: {false_positives:5d}')
    logger.info(f'  False Negatives: {false_negatives:5d}')

    # Plot confusion matrix
    logger.debug('\nGenerating confusion matrix plot')
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Ham', 'Spam'],
        yticklabels=['Ham', 'Spam'],
        cbar_kws={'label': 'Count'},
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # Add accuracy to the plot
    plt.text(
        0.5,
        -0.15,
        f'Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | '
        f'Recall: {recall:.4f} | F1-Score: {f1:.4f}',
        ha='center',
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5},
    )

    plt.tight_layout()

    # Save confusion matrix plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cm_plot_path = output_path / 'confusion_matrix.png'
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.success(f'Confusion matrix plot saved to: {cm_plot_path}')
    logger.success('\nEvaluation complete!')


if __name__ == '__main__':
    sms_spam_classifier_evaluator()
