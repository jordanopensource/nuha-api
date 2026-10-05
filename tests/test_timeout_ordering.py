"""The drain ladder: compose must outlast the engine's inner budget.

An admitted request can legitimately take up to INFERENCE_QUEUE_TIMEOUT +
INFERENCE_TIMEOUT seconds (slot wait + inference). The compose
stop_grace_period must sit ABOVE that, so a shutdown (the operator's restart
after a fetch) drains in-flight work instead of killing it mid-inference.
The engine side reads the REAL module constants; the compose side regex-parses
compose.yml (no YAML dependency), so neither number is duplicated here.
"""

import re

import app.classifier as clf
from tests.conftest import PROJECT_ROOT


_GRACE_RE = re.compile(r"stop_grace_period:\s*(\d+)s")


def test_stop_grace_exceeds_engine_budget():
    compose = (PROJECT_ROOT / "compose.yml").read_text(encoding="utf-8")
    graces = [int(m) for m in _GRACE_RE.findall(compose)]
    assert graces, "compose.yml declares no stop_grace_period"
    budget = clf.INFERENCE_QUEUE_TIMEOUT + clf.INFERENCE_TIMEOUT
    for grace in graces:
        assert grace > budget, (
            f"stop_grace_period {grace}s must exceed the engine budget "
            f"{clf.INFERENCE_QUEUE_TIMEOUT} + {clf.INFERENCE_TIMEOUT} = {budget}s, "
            f"or a shutdown kills admitted requests mid-inference"
        )
