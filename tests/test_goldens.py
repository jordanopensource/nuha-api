"""The wire-shape golden: the real app serializes results into the frozen shape.

Drives the real routes with the engine patched to return the case's synthetic
``ClassificationResult`` objects, then asserts the response body carries
exactly the contract field set, the all-null invalid shape, batch wrapping,
ordering, and a full value round-trip (a dropped or renamed field fails).
Under the split this ran twice (gateway relay + model-side serialization);
one app means one golden.
"""

from unittest.mock import patch

import pytest

from tests._goldens import CASES, assert_result_shape, endpoint_for, request_for
from tests.conftest import ALL_DIALECTS


_DIALECT = sorted(ALL_DIALECTS)[0]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_app_serializes_case(api, case):
    """One case in, the frozen wire shape out, values round-tripped."""
    from app.classifier import ClassificationResult

    results = [ClassificationResult(**r) for r in case["results"]]

    async def fake_single(entry, text, lang):
        return results[0]

    async def fake_batch(entry, texts, lang):
        return results

    with (
        patch("app.main.get_classification", side_effect=fake_single),
        patch("app.main.get_classifications_batch", side_effect=fake_batch),
    ):
        resp = api.client.post(f"/{_DIALECT}/{endpoint_for(case)}", json=request_for(case))

    assert resp.status_code == 200
    body = resp.json()
    if case["batch"]:
        assert set(body) == {"results"}
        got = body["results"]
    else:
        got = [body]
    assert len(got) == len(case["results"])
    # Shape AND value round-trip, in order: a dropped, renamed, or reordered
    # field fails here.
    for item, injected in zip(got, case["results"], strict=True):
        assert_result_shape(item)
        assert item == injected
