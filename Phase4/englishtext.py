"""
englishtext.py
--------------
English sentiment classifier. Plugs English-specific strategies into the
shared BaseTextClassifier pipeline.
"""

from nltk.stem import WordNetLemmatizer

from Phase4.languagetext import BaseTextClassifier


class EnglishClassifier(BaseTextClassifier):
    """Sentiment classifier for English text."""

    _lemmatizer = WordNetLemmatizer()  # shared, stateless — safe as class attr

    # ── language strategy hooks ──────────────────────────────────────── #

    @property
    def stopword_language(self) -> str:
        return "english"

    def is_valid_token(self, token: str) -> bool:
        """Accept only tokens made up of ASCII alphabetic characters."""
        return token.isascii() and token.isalpha()

    def stem(self, token: str) -> str:
        return self._lemmatizer.lemmatize(token)
