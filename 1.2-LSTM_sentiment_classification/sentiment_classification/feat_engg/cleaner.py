import re
from html import unescape

import pandas as pd
from loguru import logger


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate reviews from the dataset.

    Args:
        df (pd.DataFrame): The dataframe containing the reviews.
    Returns:
        pd.DataFrame: The dataframe with duplicates removed.
    """
    original_count = len(df)
    df_cleaned = df.drop_duplicates(subset=['review'], keep='first')
    duplicates_removed = original_count - len(df_cleaned)

    if duplicates_removed > 0:
        logger.info(f'Removed {duplicates_removed} duplicate reviews.')

    return df_cleaned


class ReviewCleaner:
    """A class for cleaning and preprocessing text reviews using state in/state out approach.

    This class allows chaining of cleaning operations by maintaining internal state.
    Each method returns self to enable method chaining.
    """

    def __init__(self, text: str = ''):
        """Initialize the ReviewCleaner with optional text.

        Args:
            text (str): The initial text to clean. Defaults to empty string.
        """
        self.text = text if isinstance(text, str) else ''
        # Compile regex patterns for efficiency
        self.html_tag_pattern = re.compile(r'<.*?>')
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.punctuation_pattern = re.compile(r'[^\w\s]')
        self.digit_pattern = re.compile(r'\d+')
        self.whitespace_pattern = re.compile(r'\s+')

    def set_text(self, text: str) -> 'ReviewCleaner':
        """Set the text to be cleaned.

        Args:
            text (str): The text to clean.
        Returns:
            ReviewCleaner: Self for method chaining.
        """
        self.text = text if isinstance(text, str) else ''
        return self

    def lowercase(self) -> 'ReviewCleaner':
        """Convert text to lowercase.

        Returns:
            ReviewCleaner: Self for method chaining.
        """
        self.text = self.text.lower()
        return self

    def strip_html(self) -> 'ReviewCleaner':
        """Remove HTML tags and unescape HTML entities.

        Returns:
            ReviewCleaner: Self for method chaining.
        """
        self.text = unescape(self.text)
        self.text = self.html_tag_pattern.sub('', self.text)
        return self

    def remove_urls(self) -> 'ReviewCleaner':
        """Remove URLs from text.

        Returns:
            ReviewCleaner: Self for method chaining.
        """
        self.text = self.url_pattern.sub('', self.text)
        return self

    def remove_punctuation(self) -> 'ReviewCleaner':
        """Remove punctuation from text.

        Returns:
            ReviewCleaner: Self for method chaining.
        """
        self.text = self.punctuation_pattern.sub(' ', self.text)
        return self

    def remove_digits(self) -> 'ReviewCleaner':
        """Remove digits from text.

        Returns:
            ReviewCleaner: Self for method chaining.
        """
        self.text = self.digit_pattern.sub('', self.text)
        return self

    def remove_extra_spaces(self) -> 'ReviewCleaner':
        """Remove extra whitespace from text.

        Returns:
            ReviewCleaner: Self for method chaining.
        """
        self.text = self.whitespace_pattern.sub(' ', self.text)
        self.text = self.text.strip()
        return self

    def get_text(self) -> str:
        """Get the current state of the text.

        Returns:
            str: The cleaned text.
        """
        return self.text

    def clean_text(self, text: str) -> str:
        """Convenience method to clean text in one call (maintains backward compatibility).

        Args:
            text (str): The raw text review.
        Returns:
            str: The cleaned text review.
        """
        return (
            self.set_text(text)
            .lowercase()
            .strip_html()
            .remove_urls()
            .remove_punctuation()
            .remove_digits()
            .remove_extra_spaces()
            .get_text()
        )
