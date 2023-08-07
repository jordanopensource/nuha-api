"""Interface type definitions."""

from typing import Literal, Optional
from pydantic import BaseModel  # pylint:disable=E0611


class PredictionRequest(BaseModel):
    """Single instance of comment to predict."""

    comment: str
    post: Optional[str]


class PredictionResponse(BaseModel):
    """Single instance of comment prediction"""

    label: Literal["hate-speech", "non-hate-speech"]
    score: float
    model_version: str
    comment: str
    post: str
