"""
Text classification module.

This module provides the interface between the API and the ML model.
Currently returns placeholder values until the model is integrated.
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

import emoji

_max_workers = int(os.getenv("CLASSIFIER_WORKERS", "4"))
_executor = ThreadPoolExecutor(max_workers=_max_workers)


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Result of a text classification."""

    is_valid: bool
    sub_class: str | None
    main_class: str | None
    confidence: float | None


# Placeholder values - replace with actual model classes when ready
_PLACEHOLDER_LABEL = "neutral"
_PLACEHOLDER_CONFIDENCE = 0.0


@lru_cache(maxsize=1)
def load_model():
    """Load model once, cache it."""
    # TODO: Replace with actual model loading
    # e.g., return joblib.load("model.pkl")
    # e.g., return fasttext.load_model("model.bin")
    return None


def _is_arabic(char: str) -> bool:
    """Check if the character is within the Arabic Unicode block."""
    return "\u0600" <= char <= "\u06FF"


def _clean_text(text: str) -> str:
    """
    Preprocess text before classification.

    Applies the same cleaning as the training pipeline:
    - Filters to Arabic characters, spaces, and emojis only
    - Removes very long texts (>50 words)
    - Removes emoji-only texts
    - Normalizes whitespace
    """
    # Remove very long texts
    if len(text.split()) > 50:
        return ""

    # Keep only Arabic letters, emojis, and spaces
    filtered = "".join(
        char for char in text if _is_arabic(char) or char == " " or emoji.is_emoji(char)
    )

    # Normalize whitespace
    filtered = " ".join(filtered.split()).strip()

    # Reject emoji-only texts
    if filtered and all(emoji.is_emoji(char) for char in filtered.replace(" ", "")):
        return ""

    return filtered


def _predict_single(text: str) -> ClassificationResult:
    """Synchronous single prediction."""
    cleaned = _clean_text(text)

    if not cleaned:
        return ClassificationResult(is_valid=False, sub_class=None, main_class=None, confidence=None)

    model = load_model()

    # TODO: Replace with actual inference
    # prediction = model.predict([cleaned])
    # sub_class, main_class, confidence = prediction[0], _PLACEHOLDER_LABEL, prediction[1]
    _ = model, cleaned  # Acknowledge to avoid linter warnings

    return ClassificationResult(
        is_valid=True,
        sub_class=_PLACEHOLDER_LABEL,
        main_class=_PLACEHOLDER_LABEL,
        confidence=_PLACEHOLDER_CONFIDENCE,
    )


def _predict_batch(texts: list[str]) -> list[ClassificationResult]:
    """
    Synchronous batch prediction.

    Most models are MUCH faster predicting a batch
    than predicting one-by-one in a loop.
    """
    cleaned = [_clean_text(t) for t in texts]

    # Track which indices are valid vs invalid
    valid_indices = [i for i, c in enumerate(cleaned) if c]
    valid_texts = [cleaned[i] for i in valid_indices]

    # Initialize all results as invalid
    results: list[ClassificationResult] = [
        ClassificationResult(is_valid=False, sub_class=None, main_class=None, confidence=None)
        for _ in texts
    ]

    # Only run model on valid texts
    if valid_texts:
        model = load_model()

        # TODO: Replace with actual batch inference
        # predictions = model.predict(valid_texts)
        # for i, idx in enumerate(valid_indices):
        #     results[idx] = ClassificationResult(
        #         is_valid=True,
        #         sub_class=predictions[i][0],
        #         main_class=_PLACEHOLDER_LABEL,
        #         confidence=predictions[i][1]
        #     )
        _ = model  # Acknowledge to avoid linter warnings

        for idx in valid_indices:
            results[idx] = ClassificationResult(
                is_valid=True,
                sub_class=_PLACEHOLDER_LABEL,
                main_class=_PLACEHOLDER_LABEL,
                confidence=_PLACEHOLDER_CONFIDENCE
            )

    return results


async def get_classification(text: str) -> ClassificationResult:
    """Async single classification."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _predict_single, text)


async def get_classifications_batch(texts: list[str]) -> list[ClassificationResult]:
    """Async batch classification — uses true batching, not sequential."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _predict_batch, texts)