"""Data pipeline CLI commands."""

import click

from spam_classifier.pipelines.data.cleaning import clean_sms_data
from spam_classifier.pipelines.data.inspection import inspect_labels
from spam_classifier.pipelines.data.preprocessing import data_preprocessor


@click.group()
def data():
    """Data pipeline commands for cleaning, inspection, and preprocessing."""


# Register data pipeline commands
data.add_command(clean_sms_data, name='clean')
data.add_command(inspect_labels, name='inspect')
data.add_command(data_preprocessor, name='preprocess')
