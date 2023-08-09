"""Model for Nuha."""
from typing import Literal
from transformers import AutoTokenizer
from optimum.pipelines import pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification


class Nuha:
    """Encapsulator for Nuha."""

    BATCH_SIZE = 32

    def __init__(self, model_path: str, model_version: str) -> None:
        self.model_path = model_path
        self.model_version = model_version
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path, revision=model_version
        )
        self.model = ORTModelForSequenceClassification.from_pretrained(
            model_id=model_path, revision=model_version
        )

        self.classifier = pipeline(
            task="text-classification",
            model=self.model,
            accelerator="ort",
            tokenizer=self.tokenizer,
            device=self.device,
        )

    def predict(
        self, batch: list[str]
    ) -> list[tuple[Literal["hatespeech", "non-hatespeech"], float]]:
        """Run model inference on a batch of comments.

        Returns:
            list[tuple[Literal["hatespeech", "non-hatespeech"], float]]:
            list of labels and scores for each comment
        """
        output = self.classifier(batch, batch_size=self.BATCH_SIZE)
        return [(o["label"], o["score"]) for o in output]
