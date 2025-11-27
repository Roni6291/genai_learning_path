"""Naive Bayes classifier for SMS spam detection (comparison baseline)."""

import pickle
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
from sklearn.naive_bayes import MultinomialNB


@click.command()
@click.option(
    '--train-dir',
    '-t',
    type=click.Path(exists=True),
    default='data/train',
    help='Directory containing training data (default: data/train)',
)
@click.option(
    '--test-dir',
    '-e',
    type=click.Path(exists=True),
    default='data/test',
    help='Directory containing test data (default: data/test)',
)
@click.option(
    '--output-dir',
    '-o',
    type=click.Path(),
    default='artifacts',
    help='Directory to save trained model and plots (default: artifacts)',
)
@click.option(
    '--alpha',
    '-a',
    type=float,
    default=1.0,
    help='Additive smoothing parameter (default: 1.0)',
)
def naive_bayes_classifier(
    train_dir: str,
    test_dir: str,
    output_dir: str,
    alpha: float,
) -> None:
    """Train and evaluate Naive Bayes classifier on SMS spam data.

    Uses MultinomialNB from scikit-learn as a baseline comparison
    for the neural network classifier.
    """
    logger.info('Starting Naive Bayes classifier pipeline')

    # Load training data
    train_path = Path(train_dir)
    logger.debug(f'Loading training data from {train_path}')

    x_train = np.load(train_path / 'X_train.npy')
    y_train = np.load(train_path / 'y_train.npy')

    logger.info(f'Training data: X={x_train.shape}, y={y_train.shape}')

    # Load test data
    test_path = Path(test_dir)
    logger.debug(f'Loading test data from {test_path}')

    x_test = np.load(test_path / 'X_test.npy')
    y_test = np.load(test_path / 'y_test.npy')

    logger.info(f'Test data: X={x_test.shape}, y={y_test.shape}')

    # Initialize and train Naive Bayes classifier
    logger.info(f'Training Naive Bayes classifier (alpha={alpha})')
    nb_classifier = MultinomialNB(alpha=alpha)
    nb_classifier.fit(x_train, y_train)

    logger.success('Training complete!')

    # Make predictions
    logger.info('Generating predictions on test set')
    y_pred = nb_classifier.predict(x_test)
    y_pred_proba = nb_classifier.predict_proba(x_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Display evaluation results
    logger.success('=' * 60)
    logger.success('Naive Bayes Classifier - Test Set Evaluation:')
    logger.success('=' * 60)

    logger.info('\nDetailed Classification Metrics:')
    logger.info(f'  Accuracy:  {accuracy:.4f}')
    logger.info(f'  Precision: {precision:.4f}')
    logger.info(f'  Recall:    {recall:.4f}')
    logger.info(f'  F1-Score:  {f1:.4f}')

    # Calculate confusion matrix values
    true_positives = np.sum((y_pred == 1) & (y_test == 1))
    true_negatives = np.sum((y_pred == 0) & (y_test == 0))
    false_positives = np.sum((y_pred == 1) & (y_test == 0))
    false_negatives = np.sum((y_pred == 0) & (y_test == 1))

    logger.info('\nConfusion Matrix:')
    logger.info(f'  True Positives:  {true_positives:5d}')
    logger.info(f'  True Negatives:  {true_negatives:5d}')
    logger.info(f'  False Positives: {false_positives:5d}')
    logger.info(f'  False Negatives: {false_negatives:5d}')

    logger.success('=' * 60)

    # Plot confusion matrix
    logger.info('\nGenerating confusion matrix plot')
    cm = confusion_matrix(y_test, y_pred)

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
    plt.title('Confusion Matrix - Naive Bayes', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # Add metrics to the plot
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
    cm_plot_path = output_path / 'naive_bayes_confusion_matrix.png'
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.success(f'Confusion matrix plot saved to: {cm_plot_path}')

    # Sample predictions
    logger.info('\nSample Predictions (first 10):')
    for i in range(min(10, len(y_test))):
        actual = 'SPAM' if y_test[i] == 1 else 'HAM'
        predicted = 'SPAM' if y_pred[i] == 1 else 'HAM'
        confidence = y_pred_proba[i]
        status = '✓' if y_pred[i] == y_test[i] else '✗'

        logger.info(
            f'{status} Sample {i + 1:2d}: Actual={actual:4s}, '
            f'Predicted={predicted:4s}, Confidence={confidence:.4f}'
        )

    # Save the model
    model_path = output_path / 'naive_bayes_model.pkl'
    logger.info(f'\nSaving Naive Bayes model to {model_path}')
    with open(model_path, 'wb') as f:
        pickle.dump(nb_classifier, f)

    logger.success(f'Model saved to: {model_path}')
    logger.success('\nNaive Bayes evaluation complete!')


if __name__ == '__main__':
    naive_bayes_classifier()
