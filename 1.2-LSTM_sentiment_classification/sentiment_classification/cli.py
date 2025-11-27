"""
CLI for LSTM Sentiment Classification

This module provides a command-line interface with subcommands for:
- data: Data-related operations (inspection, preprocessing)
- train: Model training operations
- eval: Model evaluation operations
"""

import click

from sentiment_classification.workflows.cleaning import clean_dataset
from sentiment_classification.workflows.embedding_creation import build_embeddings
from sentiment_classification.workflows.evaluation import evaluate_model
from sentiment_classification.workflows.inspection import inspect_dataset
from sentiment_classification.workflows.splitting import split_dataset
from sentiment_classification.workflows.tokenization import tokenize_dataset
from sentiment_classification.workflows.training import train_model


@click.group()
def cli():
    """LSTM Sentiment Classification CLI.

    A comprehensive toolkit for sentiment analysis using LSTM neural networks.
    Includes data processing, model training, and evaluation capabilities.
    """


@cli.group()
def data():
    """Data-related operations.

    Commands for inspecting, preprocessing, and managing datasets.
    """


@cli.group()
def model():
    """Model training, evaluation operations.

    Commands for training and fine-tuning sentiment classification models and
    evaluating model performance and generating metrics.
    """


# Add commands to data group
data.add_command(inspect_dataset, name='inspect')
data.add_command(clean_dataset, name='clean')
data.add_command(tokenize_dataset, name='tokenize')
data.add_command(split_dataset, name='split')
data.add_command(build_embeddings, name='embeddings')

# Add commands to model group
model.add_command(train_model, name='train')
model.add_command(evaluate_model, name='evaluate')


if __name__ == '__main__':
    cli()
