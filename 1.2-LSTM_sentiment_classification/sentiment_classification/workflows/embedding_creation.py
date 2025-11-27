"""Embedding matrix creation workflow."""

import pickle
from pathlib import Path

import click
import numpy as np
from loguru import logger

from sentiment_classification.feat_engg.embeddings import EmbeddingMatrixBuilder


@click.command()
@click.option(
    '--tokenizer-path',
    type=click.Path(exists=True, path_type=Path),
    default=Path('artifacts/tokenizer.pkl'),
    help='Path to the saved tokenizer pickle file.',
)
@click.option(
    '--word2vec-path',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to pretrained Word2Vec binary file (e.g., GoogleNews-vectors-negative300.bin).',
)
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    default=Path('artifacts'),
    help='Directory to save the embedding matrix.',
)
@click.option(
    '--embedding-dim',
    type=int,
    default=300,
    help='Dimension of word embeddings (default: 300 for Google News).',
)
@click.option(
    '--limit',
    type=int,
    default=None,
    help='Limit number of word vectors to load (for faster loading/testing).',
)
def build_embeddings(
    tokenizer_path: Path,
    word2vec_path: Path,
    output_dir: Path,
    embedding_dim: int,
    limit: int | None,
) -> None:
    """Build embedding matrix from pretrained Word2Vec embeddings.

    This command:
    1. Loads the fitted tokenizer
    2. Loads pretrained Word2Vec embeddings
    3. Builds embedding matrix mapping vocabulary to Word2Vec vectors
    4. Initializes OOV words randomly
    5. Saves embedding matrix to artifacts
    """
    # Load tokenizer
    logger.info(f'Loading tokenizer from {tokenizer_path}')
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    word_index = tokenizer.get_vocabulary()
    vocab_size = tokenizer.get_vocabulary_size()
    logger.success(f'Loaded tokenizer with vocabulary size {vocab_size}')

    # Build embedding matrix
    builder = EmbeddingMatrixBuilder(
        word2vec_path=str(word2vec_path),
        embedding_dim=embedding_dim,
        limit=limit,
    )

    builder.load_word2vec()
    embedding_matrix = builder.build_embedding_matrix(word_index, vocab_size)

    # Save embedding matrix
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'embedding_matrix.npy'
    logger.info(f'Saving embedding matrix to {output_path}')
    np.save(output_path, embedding_matrix)

    logger.success(
        f'Embedding matrix created successfully!\n'
        f'  Shape: {embedding_matrix.shape}\n'
        f'  Saved to: {output_path}\n'
        f'  Note: Embeddings will be frozen (not trainable) during training'
    )
