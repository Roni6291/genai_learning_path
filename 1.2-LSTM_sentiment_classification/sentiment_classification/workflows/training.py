"""Model training workflow."""

import pickle
from pathlib import Path

import click
import numpy as np
from loguru import logger
from tensorflow import keras  # type: ignore

from sentiment_classification.classifier import SentimentClassifier


@click.command()
@click.option(
    '--train-dir',
    type=click.Path(exists=True, path_type=Path),
    default=Path('data/train'),
    help='Directory containing training data (sequences.npy and labels.npy).',
)
@click.option(
    '--val-dir',
    type=click.Path(exists=True, path_type=Path),
    default=Path('data/validation'),
    help='Directory containing validation data (sequences.npy and labels.npy).',
)
@click.option(
    '--embedding-matrix-path',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Path to pretrained embedding matrix (.npy). If not provided, learns embeddings from scratch.',
)
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    default=Path('artifacts/models'),
    help='Directory to save the trained model.',
)
@click.option(
    '--tokenizer-path',
    type=click.Path(exists=True, path_type=Path),
    default=Path('artifacts/tokenizer.pkl'),
    help='Path to the saved tokenizer (to get vocab size automatically).',
)
@click.option(
    '--embedding-dim',
    type=int,
    default=300,
    help='Embedding dimension. Default is 300.',
)
@click.option(
    '--max-length',
    type=int,
    default=300,
    help='Maximum sequence length. Default is 300.',
)
@click.option(
    '--lstm-units',
    type=int,
    default=128,
    help='Number of LSTM units. Default is 128.',
)
@click.option(
    '--dropout-rate',
    type=float,
    default=0.5,
    help='Dropout rate. Default is 0.5.',
)
@click.option(
    '--recurrent-dropout',
    type=float,
    default=0.2,
    help='Recurrent dropout rate. Default is 0.2.',
)
@click.option(
    '--freeze-embeddings',
    is_flag=True,
    default=True,
    help='Freeze pretrained embeddings (not trainable).',
)
@click.option(
    '--epochs',
    type=int,
    default=10,
    help='Number of training epochs. Default is 10.',
)
@click.option(
    '--batch-size',
    type=int,
    default=32,
    help='Batch size. Default is 32.',
)
@click.option(
    '--learning-rate',
    type=float,
    default=0.001,
    help='Learning rate. Default is 0.001.',
)
def train_model(
    train_dir: Path,
    val_dir: Path,
    embedding_matrix_path: Path | None,
    output_dir: Path,
    tokenizer_path: Path,
    embedding_dim: int,
    max_length: int,
    lstm_units: int,
    dropout_rate: float,
    recurrent_dropout: float,
    freeze_embeddings: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    """Train a Bidirectional LSTM sentiment classifier.

    This command:
    1. Loads training and validation data
    2. Builds the Bidirectional LSTM model
    3. Optionally loads pretrained embeddings
    4. Trains the model
    5. Saves the trained model and training history
    """
    # Load tokenizer to get vocab size
    logger.info(f'Loading tokenizer from {tokenizer_path}')
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    vocab_size = tokenizer.get_vocabulary_size()
    logger.success(f'Vocabulary size: {vocab_size}')

    # Load data
    logger.info(f'Loading training data from {train_dir}')
    X_train = np.load(train_dir / 'sequences.npy')
    y_train = np.load(train_dir / 'labels.npy')
    logger.success(f'Loaded {len(X_train)} training samples')

    logger.info(f'Loading validation data from {val_dir}')
    X_val = np.load(val_dir / 'sequences.npy')
    y_val = np.load(val_dir / 'labels.npy')
    logger.success(f'Loaded {len(X_val)} validation samples')

    # Load embedding matrix if provided
    embedding_matrix = None
    if embedding_matrix_path:
        logger.info(f'Loading embedding matrix from {embedding_matrix_path}')
        embedding_matrix = np.load(embedding_matrix_path)
        logger.success(f'Loaded embedding matrix with shape {embedding_matrix.shape}')

    # Build model
    classifier = SentimentClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_length=max_length,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        recurrent_dropout=recurrent_dropout,
    )

    classifier.build_model(
        embedding_matrix=embedding_matrix,
        freeze_embeddings=freeze_embeddings,
    )

    classifier.compile_model(
        optimizer='adam',
        learning_rate=learning_rate,
    )

    # Display model summary
    classifier.summary()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # Train model
    history = classifier.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    # Save final model
    final_model_path = output_dir / 'final_model.keras'
    classifier.save(str(final_model_path))

    # Save training history
    history_path = output_dir / 'training_history.npy'
    np.save(history_path, history.history)
    logger.success(f'Training history saved to {history_path}')

    logger.success(
        f'Training complete!\n'
        f'  Best model: {output_dir / "best_model.keras"}\n'
        f'  Final model: {final_model_path}\n'
        f'  History: {history_path}'
    )
