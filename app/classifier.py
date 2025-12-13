"""
Text classification module.

This module provides the interface between the API and the ML model.
Loads a fine-tuned transformer model for Egyptian-Arabic text classification.
"""

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

import emoji
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# -----------------------------------------------------------------------------
# Configuration from environment variables
# -----------------------------------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "./model")
CLASSIFIER_WORKERS = int(os.getenv("CLASSIFIER_WORKERS", "4"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "text").lower()

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure logging based on environment variables."""
    level = getattr(logging, LOG_LEVEL, logging.INFO)

    if LOG_FORMAT == "json":
        import json as json_lib

        class JsonFormatter(logging.Formatter):
            def format(self, record):
                return json_lib.dumps({
                    "timestamp": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                })

        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    logging.basicConfig(level=level, handlers=[handler])


_setup_logging()

# -----------------------------------------------------------------------------
# Thread pool for async inference
# -----------------------------------------------------------------------------

_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    """Get or create the thread pool executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=CLASSIFIER_WORKERS)
    return _executor


def shutdown_executor() -> None:
    """Shutdown the thread pool executor gracefully."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None
        logger.info("Classifier executor shut down")


# -----------------------------------------------------------------------------
# Language support
# -----------------------------------------------------------------------------

Language = Literal["ar", "en"]


class Lang(str, Enum):
    """Supported languages for classification labels."""
    AR = "ar"
    EN = "en"


# -----------------------------------------------------------------------------
# Label definitions with translations
# -----------------------------------------------------------------------------

# Sub-class labels: Arabic (from model) -> translations
SUB_CLASS_LABELS = {
    "ar": {
        0: "محايد",
        1: "اعتراض/رفض",
        2: "السب او التنمر",
        3: "صور نمطية ضارة",
        4: "التأثيم والإتهام",
        5: "شتائم جنسية",
        6: "التحرش الجنسي اللفظي",
        7: "العنف الجنسي",
        8: "التحريض/استعداء السلطات",
        9: "التهديد",
    },
    "en": {
        0: "Neutral",
        1: "Objection/Rejection",
        2: "Insults or Bullying",
        3: "Harmful Stereotypes",
        4: "Blame and Accusation",
        5: "Sexual Insults",
        6: "Verbal Sexual Harassment",
        7: "Sexual Violence",
        8: "Incitement/Invoking Authorities",
        9: "Threats",
    },
}

# Main class labels
MAIN_CLASS_LABELS = {
    "ar": {
        0: "محايد",
        1: "اعتراض/رفض",
        2: "لغة تمييزية او مهينة",
        3: "المحتوى الجنسي",
        4: "العنف",
    },
    "en": {
        0: "Neutral",
        1: "Objection/Rejection",
        2: "Discriminatory or Offensive Language",
        3: "Sexual Content",
        4: "Violence",
    },
}

# Mapping from sub-class ID to main-class ID
SUB_TO_MAIN_INDEX = {
    0: 0,  # Neutral -> Neutral
    1: 1,  # Objection/Rejection -> Objection/Rejection
    2: 2,  # Insults or Bullying -> Discriminatory or Offensive Language
    3: 2,  # Harmful Stereotypes -> Discriminatory or Offensive Language
    4: 2,  # Blame and Accusation -> Discriminatory or Offensive Language
    5: 3,  # Sexual Insults -> Sexual Content
    6: 3,  # Verbal Sexual Harassment -> Sexual Content
    7: 4,  # Sexual Violence -> Violence
    8: 4,  # Incitement/Invoking Authorities -> Violence
    9: 4,  # Threats -> Violence
}

# Reverse mapping: Arabic sub-class label -> sub-class ID
_AR_SUBCLASS_TO_ID: dict[str, int] = {v: k for k, v in SUB_CLASS_LABELS["ar"].items()}


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Result of a text classification."""

    is_valid: bool
    sub_class: str | None
    main_class: str | None
    confidence: float | None


@dataclass
class LoadedModel:
    """Container for loaded model components."""

    model: AutoModelForSequenceClassification
    tokenizer: AutoTokenizer
    config: dict
    device: torch.device


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_model() -> LoadedModel:
    """Load model, tokenizer, and config once, cache it."""
    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        raise RuntimeError(
            f"Model not found at {model_path}. "
            f"Set MODEL_PATH environment variable or download the model first."
        )

    logger.info(f"Loading model from {model_path}")

    # Load training config
    config_path = model_path / "training_config.json"
    if not config_path.exists():
        raise RuntimeError(
            f"Training config not found at {config_path}. "
            f"Ensure training_config.json exists in the model directory."
        )

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in training config: {e}") from e

    # Validate required config keys
    if "id2label" not in config:
        raise RuntimeError("Training config missing required 'id2label' mapping")

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for inference")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    logger.info(
        f"Model loaded successfully. "
        f"Labels: {len(config['id2label'])}, Max length: {config.get('max_length', 128)}"
    )

    return LoadedModel(model=model, tokenizer=tokenizer, config=config, device=device)


# -----------------------------------------------------------------------------
# Text preprocessing
# -----------------------------------------------------------------------------

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
    if len(text.split()) > 50:
        return ""

    filtered = "".join(
        char for char in text if _is_arabic(char) or char == " " or emoji.is_emoji(char)
    )

    filtered = " ".join(filtered.split()).strip()

    if filtered and all(emoji.is_emoji(char) for char in filtered.replace(" ", "")):
        return ""

    return filtered


# -----------------------------------------------------------------------------
# Label translation
# -----------------------------------------------------------------------------

def _get_sub_class_label(sub_class_id: int, lang: Language) -> str:
    """Get sub_class label in the specified language."""
    return SUB_CLASS_LABELS[lang][sub_class_id]


def _get_main_class_label(sub_class_id: int, lang: Language) -> str:
    """Get main_class label in the specified language from sub_class id."""
    main_class_id = SUB_TO_MAIN_INDEX[sub_class_id]
    return MAIN_CLASS_LABELS[lang][main_class_id]


def _arabic_label_to_id(arabic_label: str) -> int:
    """Convert Arabic sub-class label from model config to ID."""
    if arabic_label in _AR_SUBCLASS_TO_ID:
        return _AR_SUBCLASS_TO_ID[arabic_label]
    # Fallback: try to find by checking all labels
    raise ValueError(f"Unknown Arabic label: {arabic_label}")


# -----------------------------------------------------------------------------
# Classification functions
# -----------------------------------------------------------------------------

def _predict_single(text: str, loaded: LoadedModel, lang: Language) -> ClassificationResult:
    """Synchronous single prediction."""
    cleaned = _clean_text(text)

    if not cleaned:
        return ClassificationResult(
            is_valid=False, sub_class=None, main_class=None, confidence=None
        )

    max_length = loaded.config.get("max_length", 128)

    inputs = loaded.tokenizer(
        cleaned,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    inputs = {k: v.to(loaded.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = loaded.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        confidence, predicted_id = torch.max(probs, dim=-1)

    predicted_id = predicted_id.item()
    confidence_val = confidence.item()

    sub_class = _get_sub_class_label(predicted_id, lang)
    main_class = _get_main_class_label(predicted_id, lang)

    return ClassificationResult(
        is_valid=True,
        sub_class=sub_class,
        main_class=main_class,
        confidence=round(confidence_val, 4),
    )


def _predict_batch(
    texts: list[str], loaded: LoadedModel, lang: Language
) -> list[ClassificationResult]:
    """
    Synchronous batch prediction.

    Most models are MUCH faster predicting a batch
    than predicting one-by-one in a loop.
    """
    cleaned = [_clean_text(t) for t in texts]

    valid_indices = [i for i, c in enumerate(cleaned) if c]
    valid_texts = [cleaned[i] for i in valid_indices]

    results: list[ClassificationResult] = [
        ClassificationResult(
            is_valid=False, sub_class=None, main_class=None, confidence=None
        )
        for _ in texts
    ]

    if not valid_texts:
        return results

    max_length = loaded.config.get("max_length", 128)

    inputs = loaded.tokenizer(
        valid_texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(loaded.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = loaded.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        confidences, predicted_ids = torch.max(probs, dim=-1)

    for i, idx in enumerate(valid_indices):
        pred_id = predicted_ids[i].item()
        conf = confidences[i].item()

        sub_class = _get_sub_class_label(pred_id, lang)
        main_class = _get_main_class_label(pred_id, lang)

        results[idx] = ClassificationResult(
            is_valid=True,
            sub_class=sub_class,
            main_class=main_class,
            confidence=round(conf, 4),
        )

    return results


# -----------------------------------------------------------------------------
# Async API
# -----------------------------------------------------------------------------

async def get_classification(text: str, lang: Language = "ar") -> ClassificationResult:
    """Async single classification."""
    loop = asyncio.get_running_loop()
    model = load_model()
    return await loop.run_in_executor(
        _get_executor(), _predict_single, text, model, lang
    )


async def get_classifications_batch(
    texts: list[str], lang: Language = "ar"
) -> list[ClassificationResult]:
    """Async batch classification — uses true batching, not sequential."""
    loop = asyncio.get_running_loop()
    model = load_model()
    return await loop.run_in_executor(
        _get_executor(), _predict_batch, texts, model, lang
    )
