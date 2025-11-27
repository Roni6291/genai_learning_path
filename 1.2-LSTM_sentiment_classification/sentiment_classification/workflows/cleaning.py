"""Data cleaning workflow for IMDB reviews dataset."""

from pathlib import Path

import click
import pandas as pd
from loguru import logger

from sentiment_classification.feat_engg.cleaner import ReviewCleaner, remove_duplicates


@click.command()
@click.option(
    "--data-path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/raw/IMDB Dataset.csv"),
    help="Path to the raw IMDB dataset CSV file.",
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default=Path("data/clean/imdb_reviews.csv"),
    help="Path to save the cleaned dataset CSV file.",
)
def clean_dataset(data_path: Path, output_path: Path) -> None:
    """Clean the IMDB reviews dataset and save to output path.

    This command:
    1. Loads the raw IMDB dataset
    2. Removes duplicate reviews
    3. Cleans each review (lowercase, remove HTML, URLs, punctuation, digits, extra spaces)
    4. Saves the cleaned dataset
    """
    logger.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    logger.success(f"Loaded {len(df)} reviews")

    # Remove duplicates
    logger.info("Removing duplicate reviews...")
    df = remove_duplicates(df)
    logger.success(f"Dataset now has {len(df)} unique reviews")

    # Clean reviews
    logger.info("Cleaning review text...")
    cleaner = ReviewCleaner()
    df["review"] = df["review"].apply(cleaner.clean_text)
    logger.success("All reviews cleaned")

    # Save cleaned dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving cleaned dataset to {output_path}")
    df.to_csv(output_path, index=False)
    logger.success(f"Cleaned dataset saved with {len(df)} reviews")
