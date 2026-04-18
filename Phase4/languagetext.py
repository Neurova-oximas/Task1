import pickle
import re
import string
from abc import ABC, abstractmethod

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class BaseTextClassifier(ABC):
    """
    Template-Method base for text sentiment classifiers.

    Subclasses must implement
    ─────────────────────────
    stopword_language  → str   NLTK language name for stopword removal
    is_valid_token     → bool  language-specific token filter
    stem               → str   reduce a token to its root form
    """

    _LABEL_MAP: dict[int, str] = {0: "bad", 1: "neutral", 2: "good"}

    def __init__(self, classifier_path: str, vectorizer_path: str) -> None:
        self._model = self._load_pickle(classifier_path, "Model")
        self._vectorizer = self._load_pickle(vectorizer_path, "Vectorizer")

 
    @property
    @abstractmethod
    def stopword_language(self) -> str:
        """Return the NLTK stopword corpus language name, e.g. 'arabic'."""

    @abstractmethod
    def is_valid_token(self, token: str) -> bool:
        """Return True if *token* belongs to the target language/script."""

    @abstractmethod
    def stem(self, token: str) -> str:
        """Reduce *token* to its root/stem form."""

    def classify(self, text: str) -> str:
        """Return the sentiment label for *text*."""
        processed = self._preprocess(text)
        joined = " ".join(processed)
        X = self._vectorizer.transform(pd.Series([joined]))
        prediction = self._model.predict(X)[0]
        return self._LABEL_MAP.get(int(prediction), str(prediction))

    def _preprocess(self, text: str) -> list[str]:
        tokens = word_tokenize(text)
        tokens = self._remove_stopwords(tokens)
        tokens = self._remove_punctuation(tokens)
        tokens = self._stem_tokens(tokens)
        return tokens

    def _remove_stopwords(self, tokens: list[str]) -> list[str]:
        stop = set(stopwords.words(self.stopword_language))
        return [t for t in tokens if t not in stop]

    def _remove_punctuation(self, tokens: list[str]) -> list[str]:
        extra_punct = set("،؛؟")
        return [
            t for t in tokens
            if t not in string.punctuation
            and t not in extra_punct
            and self.is_valid_token(t)
        ]

    def _stem_tokens(self, tokens: list[str]) -> list[str]:
        return [self.stem(t) for t in tokens]

    @staticmethod
    def _load_pickle(path: str, label: str) -> object:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        #print(f"{label} loaded successfully with pickle.")
        return obj
