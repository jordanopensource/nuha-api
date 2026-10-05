"""ML-stack mocking for the whole test suite.

The engine (``app.classifier``) imports ``onnxruntime`` for inference and
``AutoTokenizer`` from ``transformers``. The suite runs without the real
onnxruntime wheel or any model files, so we inject mocks for the ML stack
BEFORE any app code is imported: onnxruntime is always a MagicMock, and
transformers' deep import chain (which can fail on some hosts on a missing
native lib like ``libprotobuf.so``) is short-circuited so ``from transformers
import AutoTokenizer`` yields a MagicMock instead of triggering it.

The root ``tests/conftest.py`` calls ``install_ml_mocks()`` at import, before
anything touches ``app``; it is idempotent, so a direct call from a standalone
script is safe too.
"""

import contextlib
import sys
from unittest.mock import MagicMock


# transformers sub-modules whose real import chain can reach pyarrow/protobuf and
# fail on some hosts. Pre-seeding them with mocks keeps the real lazy import from
# ever getting that far when it resolves AutoTokenizer.
_MODULES_TO_MOCK = [
    "transformers.models.auto.modeling_auto",
    "transformers.models.auto.auto_factory",
    "transformers.generation",
    "transformers.generation.utils",
    "transformers.generation.candidate_generator",
]

_installed = False


def install_ml_mocks() -> None:
    """Mock onnxruntime + transformers in sys.modules (idempotent)."""
    global _installed
    if _installed:
        return
    _installed = True

    # onnxruntime is always mocked: no native runtime is needed for the
    # contract/logic tests, and the wheel is not installed in the test env.
    sys.modules["onnxruntime"] = MagicMock()

    for name in _MODULES_TO_MOCK:
        if name not in sys.modules:
            sys.modules[name] = MagicMock()

    # Make `from transformers import AutoTokenizer` resolve to a MagicMock. Prefer
    # the real lazy package (so its __getattr__ can be wrapped); if transformers is
    # not installed at all, fall back to a fully mocked top-level module.
    try:
        import transformers as tf
    except Exception:  # transformers not installed on this host
        tf = MagicMock()
        sys.modules["transformers"] = tf

    if not getattr(tf, "_test_patched", False):
        orig_getattr = getattr(type(tf), "__getattr__", None)

        def safe_getattr(self, name):
            """Return a MagicMock for tokenizer/model classes instead of triggering
            transformers' deep import chain."""
            if name in (
                "AutoTokenizer",
                "PreTrainedModel",
                "PreTrainedTokenizer",
                "PreTrainedTokenizerFast",
            ):
                return MagicMock()
            if orig_getattr is not None:
                return orig_getattr(self, name)
            raise AttributeError(name)

        # MagicMock instances accept attribute assignment but have no settable
        # __getattr__ on the type; guard so the real-package path still patches.
        with contextlib.suppress(TypeError, AttributeError):
            type(tf).__getattr__ = safe_getattr
        tf._test_patched = True
