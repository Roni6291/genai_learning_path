"""Text cleaning module for spam classifier.

Provides TextCleaner class for preprocessing text data.
"""

import contextlib
import re
import string

import nltk
from loguru import logger
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


class TextCleaner:
    """A class for cleaning and preprocessing text data.

    Uses a state in/state out approach where the class object maintains state.
    """

    def __init__(self):
        """Initialize the TextCleaner with necessary NLTK resources."""
        logger.debug("Initializing TextCleaner")
        self.text = ''
        self.tokens = []
        self.lemmatizer = WordNetLemmatizer()

        # Download required NLTK data
        self._download_nltk_resources()

        # Load stopwords
        self.stop_words = set(stopwords.words('english'))
        logger.debug("TextCleaner initialized successfully")

    def _download_nltk_resources(self):
        """Download required NLTK resources if not already present."""
        resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                logger.info(f"Downloading NLTK resource: {resource}")
                with contextlib.suppress(Exception):
                    nltk.download(resource, quiet=True)

    def reset(self) -> 'TextCleaner':
        """Reset the cleaner state to process new text.

        Clears text and tokens, allowing reuse of the same instance.

        Returns:
            self: Returns the instance for method chaining
        """
        self.text = ''
        self.tokens = []
        return self

    def load_text(self, text: str) -> 'TextCleaner':
        """Load text into the cleaner.

        Args:
            text: Input text to clean

        Returns:
            self: Returns the instance for method chaining
        """
        self.text = text
        return self

    def lowercase(self) -> 'TextCleaner':
        """Convert text to lowercase.

        Returns:
            self: Returns the instance for method chaining
        """
        self.text = self.text.lower()
        return self

    def remove_urls(self) -> 'TextCleaner':
        """Remove URLs from text.

        Returns:
            self: Returns the instance for method chaining
        """
        # Remove URLs starting with http, https, www
        url_pattern = r'https?://\S+|www\.\S+'
        self.text = re.sub(url_pattern, '', self.text)
        return self

    def remove_digits(self) -> 'TextCleaner':
        """Remove all digits from text.

        Returns:
            self: Returns the instance for method chaining
        """
        self.text = re.sub(r'\d+', '', self.text)
        return self

    def remove_punctuation(self) -> 'TextCleaner':
        """Remove punctuation from text.

        Returns:
            self: Returns the instance for method chaining
        """
        self.text = self.text.translate(str.maketrans('', '', string.punctuation))
        return self

    def remove_special_characters(self) -> 'TextCleaner':
        """Remove special characters, keeping only letters and spaces.

        Returns:
            self: Returns the instance for method chaining
        """
        self.text = re.sub(r'[^a-zA-Z\s]', '', self.text)
        return self

    def tokenize(self) -> 'TextCleaner':
        """Tokenize the text into words.

        Returns:
            self: Returns the instance for method chaining
        """
        self.tokens = word_tokenize(self.text)
        return self

    def remove_stopwords(self) -> 'TextCleaner':
        """Remove stopwords from tokens.

        Returns:
            self: Returns the instance for method chaining
        """
        self.tokens = [token for token in self.tokens if token not in self.stop_words]
        return self

    def lemmatize(self) -> 'TextCleaner':
        """Lemmatize tokens to their base form.

        Returns:
            self: Returns the instance for method chaining
        """
        self.tokens = [self.lemmatizer.lemmatize(token) for token in self.tokens]
        return self

    def get_text(self) -> str:
        """Get the current text state.

        Returns:
            str: The current text
        """
        return self.text

    def get_tokens(self) -> list[str]:
        """Get the current tokens.

        Returns:
            list[str]: The current list of tokens
        """
        return self.tokens

    def get_cleaned_text(self) -> str:
        """Get the cleaned text by joining tokens.

        Returns:
            str: Cleaned text as a single string
        """
        return ' '.join(self.tokens)

    def clean(self, text: str) -> str:
        """Complete cleaning pipeline in one method call.

        Args:
            text: Input text to clean

        Returns:
            str: Cleaned text
        """
        return (
            self.load_text(text)
            .lowercase()
            .remove_urls()
            .remove_digits()
            .remove_punctuation()
            .remove_special_characters()
            .tokenize()
            .remove_stopwords()
            .lemmatize()
            .get_cleaned_text()
        )


def clean_text(text: str, cleaner: TextCleaner | None = None) -> str:
    """Convenience function to clean text using TextCleaner.

    Args:
        text: Input text to clean
        cleaner: Optional TextCleaner instance to reuse. If None, creates a new one.

    Returns:
        str: Cleaned text
    """
    if cleaner is None:
        cleaner = TextCleaner()
    else:
        cleaner.reset()
    return cleaner.clean(text)
