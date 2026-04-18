"""
arabictext.py
-------------
Arabic sentiment classifier. Plugs Arabic-specific strategies into the
shared BaseTextClassifier pipeline.
"""

from nltk.stem.isri import ISRIStemmer

from Phase4.languagetext import BaseTextClassifier


class ArabicClassifier(BaseTextClassifier):
    """Sentiment classifier for Arabic text."""

    _stemmer = ISRIStemmer()          # shared, stateless — safe as class attr

    # ── language strategy hooks ──────────────────────────────────────── #

    @property
    def stopword_language(self) -> str:
        return "arabic"

    def is_valid_token(self, token: str) -> bool:
        """Accept only tokens that contain at least one Arabic character."""
        return bool(__import__("re").search(r"[\u0600-\u06FF]", token))

    def stem(self, token: str) -> str:
        return self._stemmer.stem(token)
