"""Structural cases for the wire-shape golden test.

The app must serialize an internal classification result into the frozen wire
shape. These cases pin the STRUCTURE only: the contract field set (taken from
the shared ClassifyResponse schema, never a hardcoded key list), the all-null
shape of an invalid input, and batch wrapping + ordering. No label string or
confidence is asserted by value. Real label strings live in the dialect files
and real confidences come from the model, so nothing model-derived is baked in
here; the synthetic results below exist only to be round-tripped through the
app and are checked by type and shape, never by equality to a fixed
'expected output'.
"""

from app.common.schemas import ClassifyResponse


# The frozen contract field set, from the shared response schema (not a hardcoded
# key list): {is_valid, sub_class, main_class, confidence}.
CONTRACT_FIELDS = frozenset(ClassifyResponse.model_fields)


def _valid(tag: str) -> dict:
    """A synthetic VALID result. The sub/main strings are obvious test markers
    (not real labels) and the confidence is a mid-range probability; only their
    type and round-trip are asserted, never the specific value."""
    return {
        "is_valid": True,
        "sub_class": f"sub-{tag}",
        "main_class": f"main-{tag}",
        "confidence": 0.5,
    }


_INVALID = {"is_valid": False, "sub_class": None, "main_class": None, "confidence": None}


# The structural cases the app must serialize identically release over
# release. ``results`` are the synthetic internal results the test injects into
# inference; ``batch`` selects the endpoint and the response wrapping.
CASES = [
    {"name": "single_valid", "batch": False, "results": [_valid("a")]},
    {"name": "single_invalid", "batch": False, "results": [_INVALID]},
    {"name": "batch_mixed", "batch": True, "results": [_valid("a"), _INVALID, _valid("b")]},
]


def endpoint_for(case: dict) -> str:
    """The public path segment for a case (single vs batch)."""
    return "classify/batch" if case["batch"] else "classify"


def request_for(case: dict) -> dict:
    """A synthetic public request body for a case. Inference is mocked, so the
    text content is irrelevant; only the single-vs-batch shape and the batch
    length matter."""
    if case["batch"]:
        return {"texts": ["text" for _ in case["results"]]}
    return {"text": "text"}


def assert_result_shape(item: dict) -> None:
    """Assert one result obeys the frozen wire contract: exactly the contract
    fields, and either a valid triple (non-empty str, non-empty str, probability)
    or the all-null invalid shape. Checks type/shape, not specific values."""
    assert frozenset(item) == CONTRACT_FIELDS, item
    if item["is_valid"]:
        assert isinstance(item["sub_class"], str) and item["sub_class"]
        assert isinstance(item["main_class"], str) and item["main_class"]
        assert isinstance(item["confidence"], float) and 0.0 <= item["confidence"] <= 1.0
    else:
        assert item["sub_class"] is None
        assert item["main_class"] is None
        assert item["confidence"] is None
