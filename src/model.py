"""Model for Nuha."""
from dataclasses import dataclass
from typing import Literal
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.pipelines import pipeline
from transformers import AutoTokenizer


@dataclass
class PredictionResult:
    """Model prediction result."""

    label: Literal["offensive-language", "not-online-violence", "gender-based-violence"]
    score: float


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

    def predict(self, batch: list[str]) -> list[PredictionResult]:
        """Run model inference on a batch of comments.

        Returns:
            list[PredictionResult]: list of labels and scores for each comment
        """
        output = self.classifier(batch, batch_size=self.BATCH_SIZE)
        print(output)
        return [
            PredictionResult(
                label=o["label"].lower().replace(" ", "-"), score=o["score"]
            )
            for o in output
        ]
