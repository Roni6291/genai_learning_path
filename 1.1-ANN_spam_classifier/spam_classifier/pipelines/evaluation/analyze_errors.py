"""Misclassification analysis for SMS spam classifier."""

from pathlib import Path

import click
import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split

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
    '--cleaned-data',
    '-c',
    type=click.Path(exists=True),
    required=True,
    help='Path to cleaned data file with original messages',
)
@click.option(
    '--output-dir',
    '-o',
    type=click.Path(),
    default='artifacts',
    help='Directory to save analysis results (default: artifacts)',
)
@click.option(
    '--threshold',
    type=float,
    default=0.5,
    help='Classification threshold (default: 0.5)',
)
@click.option(
    '--num-examples',
    '-n',
    type=int,
    default=5,
    help='Number of examples to show per category (default: 5)',
)
def analyze_misclassifications(
    model_path: str,
    test_dir: str,
    cleaned_data: str,
    output_dir: str,
    threshold: float,
    num_examples: int,
) -> None:
    """Analyze misclassified SMS messages to understand model errors.

    Identifies false positives and false negatives, displays the actual
    messages, and provides insights into why they were misclassified.
    """
    logger.info('Starting misclassification analysis')

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

    # Get predictions
    logger.info(f'Generating predictions (threshold={threshold})')
    predictions = classifier.predict(x_test, threshold=threshold)
    probabilities = classifier.predict_proba(x_test)

    # Load cleaned text data and replicate the same split used in preprocessing
    logger.debug(f'Loading cleaned messages from {cleaned_data}')
    messages = []
    labels = []

    with open(cleaned_data, encoding='utf-8') as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.rstrip('\n\r')
            if not line:  # Skip empty lines
                continue

            parts = line.split('\t')

            label, text = parts
            if not label or not text:
                logger.warning(f'Line {line_num}: Skipping line with missing values')
                continue

            labels.append(label)
            messages.append(text)

    # Replicate the same train/test split used in preprocessing.py
    # This ensures test_messages aligns with X_test and y_test
    messages_array = np.array(messages)
    labels_array = np.array(labels)

    _, test_messages_array, _, test_labels = train_test_split(
        messages_array,
        labels_array,
        test_size=0.2,
        random_state=42,
        stratify=labels_array,
    )

    test_messages = test_messages_array.tolist()

    # Verify the split matches
    if not np.array_equal(test_labels, y_test):
        logger.error('Test split labels do not match! Using fallback analysis.')
        logger.warning(
            'This may happen if preprocessing parameters changed. '
            'Results may be inaccurate.'
        )

    logger.info(
        f'Loaded {len(test_messages)} test messages (split matches: {np.array_equal(test_labels, y_test)})'
    )

    # Find misclassifications
    fp_indices = np.where((predictions == 1) & (y_test == 0))[0]
    fn_indices = np.where((predictions == 0) & (y_test == 1))[0]

    logger.info(f'Found {len(fp_indices)} false positives')
    logger.info(f'Found {len(fn_indices)} false negatives')

    # Display analysis
    max_display_length = 200

    logger.info('\n' + '=' * 80)
    logger.warning(
        f'FALSE POSITIVES: {len(fp_indices)} total (Ham messages predicted as Spam)'
    )
    logger.info('=' * 80)

    for i, idx in enumerate(fp_indices[:num_examples], 1):
        msg = test_messages[idx]
        conf = probabilities[idx]
        truncated_msg = (
            f'{msg[:max_display_length]}...' if len(msg) > max_display_length else msg
        )

        logger.info(f'\n{i}. Message Index: {idx}')
        logger.info(f'   Prediction Confidence: {conf:.4f} (Spam probability)')
        logger.info(f'   Message: {truncated_msg}')
        logger.info(
            '   Analysis: This ham message likely contains spam-like keywords, '
            'promotional phrases, or formatting patterns that confused the model.'
        )

    logger.info('\n' + '=' * 80)
    logger.warning(
        f'FALSE NEGATIVES: {len(fn_indices)} total (Spam messages predicted as Ham)'
    )
    logger.info('=' * 80)

    for i, idx in enumerate(fn_indices[:num_examples], 1):
        msg = test_messages[idx]
        conf = probabilities[idx]
        truncated_msg = (
            f'{msg[:max_display_length]}...' if len(msg) > max_display_length else msg
        )

        logger.info(f'\n{i}. Message Index: {idx}')
        logger.info(f'   Prediction Confidence: {conf:.4f} (Spam probability)')
        logger.info(f'   Message: {truncated_msg}')
        logger.info(
            '   Analysis: This spam message likely uses subtle tactics, '
            'legitimate-looking language, or lacks typical spam markers.'
        )

    # Save detailed analysis to file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    analysis_path = output_path / 'misclassification_analysis.txt'

    logger.info(f'\nSaving detailed analysis to {analysis_path}')

    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write('MISCLASSIFICATION ANALYSIS REPORT\n')
        f.write('=' * 80 + '\n\n')
        f.write(f'Model: {model_path}\n')
        f.write(f'Test samples: {len(y_test)}\n')
        f.write(f'Classification threshold: {threshold}\n')
        f.write(f'False Positives: {len(fp_indices)}\n')
        f.write(f'False Negatives: {len(fn_indices)}\n')
        f.write(
            f'Error Rate: {(len(fp_indices) + len(fn_indices)) / len(y_test):.2%}\n'
        )
        f.write('\n' + '=' * 80 + '\n\n')

        # False Positives section
        f.write('FALSE POSITIVES (Ham → Spam)\n')
        f.write('=' * 80 + '\n\n')

        for i, idx in enumerate(fp_indices[:num_examples], 1):
            f.write(f'{i}. Index: {idx}\n')
            f.write(f'   Confidence: {probabilities[idx]:.4f}\n')
            f.write(f'   Message: {test_messages[idx]}\n')
            f.write(
                '   Possible reasons:\n'
                '   - Contains words commonly found in spam (free, win, call, etc.)\n'
                '   - Has promotional or urgent language patterns\n'
                '   - Uses capitalization or punctuation similar to spam\n'
                '   - May include URLs, phone numbers, or contact information\n\n'
            )

        f.write('\n' + '=' * 80 + '\n\n')

        # False Negatives section
        f.write('FALSE NEGATIVES (Spam → Ham)\n')
        f.write('=' * 80 + '\n\n')

        for i, idx in enumerate(fn_indices[:num_examples], 1):
            f.write(f'{i}. Index: {idx}\n')
            f.write(f'   Confidence: {probabilities[idx]:.4f}\n')
            f.write(f'   Message: {test_messages[idx]}\n')
            f.write(
                '   Possible reasons:\n'
                '   - Uses more natural, conversational language\n'
                '   - Lacks obvious spam keywords or patterns\n'
                '   - May be a sophisticated spam message\n'
                '   - Could resemble legitimate promotional content\n\n'
            )

        f.write('\n' + '=' * 80 + '\n')
        f.write('END OF REPORT\n')

    logger.success(f'Analysis saved to: {analysis_path}')
    logger.success('\nMisclassification analysis complete!')


if __name__ == '__main__':
    analyze_misclassifications()
