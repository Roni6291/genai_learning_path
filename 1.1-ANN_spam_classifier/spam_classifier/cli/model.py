"""Model training and evaluation CLI commands."""

import click

from spam_classifier.pipelines.evaluation.analyze_errors import (
    analyze_misclassifications,
)
from spam_classifier.pipelines.evaluation.eval import sms_spam_classifier_evaluator
from spam_classifier.pipelines.train.training import train_classifier


@click.group()
def model():
    """Model training and evaluation commands."""


# Register model pipeline commands
model.add_command(train_classifier, name='train')
model.add_command(sms_spam_classifier_evaluator, name='evaluate')
model.add_command(analyze_misclassifications, name='analyze-errors')
