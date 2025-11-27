"""Comparison and baseline model CLI commands."""

import click

from spam_classifier.pipelines.compare.naive_bayes_classification import (
    naive_bayes_classifier,
)


@click.group()
def compare():
    """Baseline and comparison model commands."""


# Register comparison model commands
compare.add_command(naive_bayes_classifier, name='naive-bayes')
