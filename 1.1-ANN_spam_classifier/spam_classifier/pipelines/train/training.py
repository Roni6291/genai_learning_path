"""Training pipeline for SMS spam classifier."""

from pathlib import Path

import click
import numpy as np
from loguru import logger

from spam_classifier.classifier import SMSSpamClassifier


@click.command()
@click.option(
    '--train-dir',
    '-t',
    type=click.Path(exists=True),
    default='data/train',
    help='Directory containing training data (default: data/train)',
)
@click.option(
    '--output-dir',
    '-o',
    type=click.Path(),
    default='artifacts',
    help='Directory to save trained model (default: artifacts)',
)
@click.option(
    '--epochs',
    type=int,
    default=10,
    help='Number of training epochs (default: 10)',
)
@click.option(
    '--batch-size',
    '-b',
    type=int,
    default=32,
    help='Batch size for training (default: 32)',
)
@click.option(
    '--hidden-layers',
    '-l',
    type=str,
    default='128,64',
    help='Hidden layer sizes, comma-separated (default: 128,64)',
)
@click.option(
    '--dropout',
    '-d',
    type=float,
    default=0.3,
    help='Dropout rate (default: 0.3)',
)
@click.option(
    '--learning-rate',
    '-lr',
    type=float,
    default=0.001,
    help='Learning rate (default: 0.001)',
)
@click.option(
    '--validation-split',
    '-v',
    type=float,
    default=0.2,
    help='Validation split from training data (default: 0.2)',
)
def train_classifier(
    train_dir: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    hidden_layers: str,
    dropout: float,
    learning_rate: float,
    validation_split: float,
) -> None:
    """Train SMS spam classifier using preprocessed data.

    Loads preprocessed training data, trains a neural network classifier,
    and saves the trained model.
    """
    logger.info('Starting SMS spam classifier training pipeline')

    # Parse hidden layers
    layer_sizes = [int(size.strip()) for size in hidden_layers.split(',')]
    logger.debug(f'Hidden layer architecture: {layer_sizes}')

    # Load training data
    train_path = Path(train_dir)
    logger.debug(f'Loading training data from {train_path}')

    x_train = np.load(train_path / 'X_train.npy')
    y_train = np.load(train_path / 'y_train.npy')

    logger.info(f'Training data: X={x_train.shape}, y={y_train.shape}')

    # Split training data for validation
    if validation_split > 0:
        split_idx = int(len(x_train) * (1 - validation_split))
        x_val = x_train[split_idx:]
        y_val = y_train[split_idx:]
        x_train = x_train[:split_idx]
        y_train = y_train[:split_idx]

        logger.info(f'Validation data: X={x_val.shape}, y={y_val.shape}')
        logger.info(f'Adjusted training data: X={x_train.shape}, y={y_train.shape}')
    else:
        x_val = None
        y_val = None

    # Initialize classifier
    logger.info('Initializing SMS spam classifier')
    input_dim = x_train.shape[1]

    classifier = SMSSpamClassifier(
        input_dim=input_dim,
        hidden_layers=layer_sizes,
        dropout_rate=dropout,
        learning_rate=learning_rate,
    )

    # Display model architecture
    logger.info('Model architecture:')
    classifier.summary()

    # Train the model
    logger.info(f'Training model for {epochs} epochs with batch size {batch_size}')
    history = classifier.train(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # Save the model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / 'spam_classifier_model.keras'
    logger.info(f'Saving trained model to {model_path}')
    classifier.save(str(model_path))

    # Plot and save training history
    plot_path = output_path / 'training_history.png'
    logger.info(f'Plotting training history to {plot_path}')
    classifier.plot_training_history(
        history=history, save_path=str(plot_path), show_plot=False
    )

    logger.success(f'Training complete! Model saved to {model_path}')

    # Log training summary
    if classifier.model is not None:
        logger.info(f'  Total parameters: {classifier.model.count_params():,}')

    # Get final training metrics
    final_epoch = len(history.history['loss']) - 1
    logger.info(f'  Final training loss: {history.history["loss"][final_epoch]:.4f}')
    logger.info(
        f'  Final training accuracy: {history.history["accuracy"][final_epoch]:.4f}'
    )

    if x_val is not None:
        logger.info(
            f'  Final validation loss: {history.history["val_loss"][final_epoch]:.4f}'
        )
        logger.info(
            f'  Final validation accuracy: {history.history["val_accuracy"][final_epoch]:.4f}'
        )

    logger.info('=' * 60)


if __name__ == '__main__':
    train_classifier()
