"""Tokenization workflow for IMDB reviews dataset."""

import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger

from sentiment_classification.feat_engg.tokenizer import ReviewTokenizer


@click.command()
@click.option(
    '--data-path',
    type=click.Path(exists=True, path_type=Path),
    default=Path('data/clean/imdb_reviews.csv'),
    help='Path to the cleaned IMDB dataset CSV file.',
)
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    default=Path('data/processed'),
    help='Directory to save the tokenized data.',
)
@click.option(
    '--max-words',
    type=int,
    default=10000,
    help='Maximum number of words to keep in vocabulary.',
)
@click.option(
    '--max-len',
    type=int,
    default=300,
    help='Maximum length of sequences (pad or truncate to this length).',
)
@click.option(
    '--padding',
    type=click.Choice(['pre', 'post'], case_sensitive=False),
    default='post',
    help="Padding strategy: 'pre' or 'post'.",
)
@click.option(
    '--truncating',
    type=click.Choice(['pre', 'post'], case_sensitive=False),
    default='post',
    help="Truncating strategy: 'pre' or 'post'.",
)
def tokenize_dataset(
    data_path: Path,
    output_dir: Path,
    max_words: int,
    max_len: int,
    padding: str,
    truncating: str,
) -> None:
    """Tokenize the cleaned IMDB reviews dataset.

    This command:
    1. Loads the cleaned dataset
    2. Tokenizes the reviews and builds vocabulary
    3. Converts texts to sequences of integers
    4. Pads/truncates sequences to fixed length
    5. Saves tokenized data and tokenizer
    """
    # Load dataset
    logger.info(f'Loading cleaned dataset from {data_path}')
    df = pd.read_csv(data_path)
    logger.success(f'Loaded {len(df)} reviews')

    # Extract texts and labels (convert sentiment to binary: positive=1, negative=0)
    texts = df['review'].tolist()
    labels = (df['sentiment'] == 'positive').astype(int).to_numpy()  # All reviews included

    # Initialize and fit tokenizer
    logger.info(f'Building vocabulary with max {max_words} words...')
    tokenizer = ReviewTokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit(texts)

    vocab_size = tokenizer.get_vocabulary_size()
    logger.success(f'Vocabulary built with {vocab_size} unique words')

    # Convert texts to sequences and pad
    logger.info(f'Converting texts to sequences and padding to length {max_len}...')
    tokenizer.texts_to_sequences().pad_sequences(
        maxlen=max_len, padding=padding, truncating=truncating
    )

    padded_sequences = tokenizer.get_padded_sequences()
    logger.success(f'Sequences created with shape {padded_sequences.shape}')

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save tokenized sequences
    sequences_path = output_dir / 'sequences.npy'
    logger.info(f'Saving sequences to {sequences_path}')
    np.save(sequences_path, padded_sequences)

    # Save labels
    labels_path = output_dir / 'labels.npy'
    logger.info(f'Saving labels to {labels_path}')
    np.save(labels_path, labels)

    # Save tokenizer
    tokenizer_path = Path('artifacts') / 'tokenizer.pkl'
    logger.info(f'Saving tokenizer to {tokenizer_path}')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)

    logger.success(
        f'Tokenization complete! Saved to {output_dir}\n'
        f'  - Sequences: {sequences_path}\n'
        f'  - Labels: {labels_path}\n'
        f'  - Tokenizer: {tokenizer_path}\n'
    )
