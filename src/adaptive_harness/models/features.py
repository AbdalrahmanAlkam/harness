"""Feature extraction pipeline combining word and character n-grams."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_text_feature_union() -> FeatureUnion:
    """Constructs a hybrid word and character n-gram TF-IDF feature pipeline."""
    return FeatureUnion(
        transformer_list=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=4000,
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=6000,
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
