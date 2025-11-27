"""CLI package for spam classifier."""

import click

from spam_classifier.cli.compare import compare
from spam_classifier.cli.data import data
from spam_classifier.cli.model import model


@click.group()
def cli():
    """Spam Classifier CLI - Manage data pipelines and model training."""


# Register command groups
cli.add_command(data)
cli.add_command(model)
cli.add_command(compare)

__all__ = ['cli']
