"""Data splitting workflow for train/validation/test sets."""

from pathlib import Path

import click
import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split


@click.command()
@click.option(
    '--sequences-path',
    type=click.Path(exists=True, path_type=Path),
    default=Path('data/processed/sequences.npy'),
    help='Path to the sequences numpy file.',
)
@click.option(
    '--labels-path',
    type=click.Path(exists=True, path_type=Path),
    default=Path('data/processed/labels.npy'),
    help='Path to the labels numpy file.',
)
@click.option(
    '--output-base-dir',
    type=click.Path(path_type=Path),
    default=Path('data'),
    help='Base directory for output (will create train/validation/test subdirs).',
)
@click.option(
    '--train-split',
    type=float,
    default=0.8,
    help='Proportion of data for training (default: 0.8).',
)
@click.option(
    '--val-split',
    type=float,
    default=0.1,
    help='Proportion of data for validation (default: 0.1).',
)
@click.option(
    '--test-split',
    type=float,
    default=0.1,
    help='Proportion of data for testing (default: 0.1).',
)
@click.option(
    '--random-seed',
    type=int,
    default=42,
    help='Random seed for reproducibility.',
)
def split_dataset(
    sequences_path: Path,
    labels_path: Path,
    output_base_dir: Path,
    train_split: float,
    val_split: float,
    test_split: float,
    random_seed: int,
) -> None:
    """Split sequences and labels into train, validation, and test sets.

    This command:
    1. Loads sequences and labels from numpy files
    2. Splits data into train/validation/test sets
    3. Saves each split to separate directories
    """
    # Validate splits sum to 1.0
    total_split = train_split + val_split + test_split
    if not np.isclose(total_split, 1.0):
        raise ValueError(
            f'Splits must sum to 1.0, got {total_split}. '
            f'train={train_split}, val={val_split}, test={test_split}'
        )

    # Load data
    logger.info(f'Loading sequences from {sequences_path}')
    sequences = np.load(sequences_path)
    logger.info(f'Loading labels from {labels_path}')
    labels = np.load(labels_path)

    logger.success(f'Loaded {len(sequences)} samples')

    # First split: separate test set
    test_size = test_split / (train_split + val_split + test_split)
    X_temp, X_test, y_temp, y_test = train_test_split(
        sequences,
        labels,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels,
    )

    logger.info(f'Test set: {len(X_test)} samples ({test_split * 100:.1f}%)')

    # Second split: separate validation from training
    val_size = val_split / (train_split + val_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_seed, stratify=y_temp
    )

    logger.info(f'Train set: {len(X_train)} samples ({train_split * 100:.1f}%)')
    logger.info(f'Validation set: {len(X_val)} samples ({val_split * 100:.1f}%)')

    # Create output directories
    train_dir = output_base_dir / 'train'
    val_dir = output_base_dir / 'validation'
    test_dir = output_base_dir / 'test'

    for directory in [train_dir, val_dir, test_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Save training set
    logger.info(f'Saving training set to {train_dir}')
    np.save(train_dir / 'sequences.npy', X_train)
    np.save(train_dir / 'labels.npy', y_train)

    # Save validation set
    logger.info(f'Saving validation set to {val_dir}')
    np.save(val_dir / 'sequences.npy', X_val)
    np.save(val_dir / 'labels.npy', y_val)

    # Save test set
    logger.info(f'Saving test set to {test_dir}')
    np.save(test_dir / 'sequences.npy', X_test)
    np.save(test_dir / 'labels.npy', y_test)

    # Log summary
    logger.success(
        f'Data split complete!\n'
        f'  Train: {len(X_train)} samples ({len(X_train) / len(sequences) * 100:.1f}%) -> {train_dir}\n'
        f'  Validation: {len(X_val)} samples ({len(X_val) / len(sequences) * 100:.1f}%) -> {val_dir}\n'
        f'  Test: {len(X_test)} samples ({len(X_test) / len(sequences) * 100:.1f}%) -> {test_dir}'
    )
