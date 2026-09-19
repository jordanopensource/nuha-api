"""Every dialect's hf_repo must exist, be public, and ship an ONNX model.

This is the one part of the suite that reaches the network, and it runs by
default (it is a real gate, not opt-in): a dialect file pointing at a missing,
private, or non-ONNX repo is a hard failure here, caught in the suite instead
of at install time when the fetch command tries to download it.

The ONLY thing that skips these tests is genuine network unreachability (no
route to HuggingFace at all); a transient outage must not turn image builds
red, and CI/build environments that have HF access (they download the models)
will exercise the check for real. A reachable-but-wrong repo always fails.
"""

import json
import urllib.error
import urllib.request

import pytest

from tests.conftest import ALL_DIALECTS, _dialect_file


_HF_API = "https://huggingface.co/api/models/"
_TIMEOUT = 30


def _hf_repo(dialect: str) -> str:
    return json.loads(_dialect_file(dialect).read_text(encoding="utf-8"))["hf_repo"]


def _fetch_model_info(repo: str) -> dict:
    """Anonymous HF Hub API lookup for a repo.

    Returns the parsed model info on 200 (exists and is public, no token used).
    Raises urllib.error.HTTPError for 4xx/5xx (missing/private/gated), which the
    caller turns into a hard failure. Raises urllib.error.URLError (wrapping a
    socket error) only when HuggingFace is unreachable, which the caller treats
    as skip-worthy network unavailability, not a repo problem.
    """
    req = urllib.request.Request(
        f"{_HF_API}{repo}", headers={"User-Agent": "nuha-api-dialect-check"}
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode())


def _model_info_or_skip(repo: str) -> dict:
    """Fetch model info, converting only true network-unreachability into a skip.

    An HTTPError (the server answered) is a real signal about the repo and is
    left to the caller to assert on; a bare URLError/socket timeout means we
    couldn't reach HF at all, so we skip rather than fail the build.
    """
    try:
        return _fetch_model_info(repo)
    except urllib.error.HTTPError:
        raise  # server answered (404/401/403/5xx) -> real, let the test assert
    except (urllib.error.URLError, TimeoutError) as e:
        pytest.skip(f"HuggingFace unreachable ({e}); skipping network model check")


@pytest.mark.parametrize("dialect", ALL_DIALECTS)
def test_hf_repo_exists_and_is_public(dialect):
    """The repo resolves anonymously: it exists and needs no auth."""
    repo = _hf_repo(dialect)
    try:
        info = _model_info_or_skip(repo)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            pytest.fail(f"Dialect '{dialect}': hf_repo '{repo}' is private/gated (HTTP {e.code})")
        if e.code == 404:
            pytest.fail(f"Dialect '{dialect}': hf_repo '{repo}' does not exist (HTTP 404)")
        pytest.fail(f"Dialect '{dialect}': hf_repo '{repo}' lookup failed (HTTP {e.code})")
    assert not info.get("private", False), f"Dialect '{dialect}': hf_repo '{repo}' is private"


@pytest.mark.parametrize("dialect", ALL_DIALECTS)
def test_hf_repo_contains_onnx_model(dialect):
    """The repo ships at least one .onnx file (the app loads via onnxruntime,
    not PyTorch/safetensors), matching what _find_onnx_file expects at runtime."""
    repo = _hf_repo(dialect)
    try:
        info = _model_info_or_skip(repo)
    except urllib.error.HTTPError as e:
        pytest.fail(f"Dialect '{dialect}': hf_repo '{repo}' lookup failed (HTTP {e.code})")
    files = [s.get("rfilename", "") for s in info.get("siblings", [])]
    onnx_files = [f for f in files if f.endswith(".onnx")]
    assert onnx_files, (
        f"Dialect '{dialect}': hf_repo '{repo}' has no .onnx file (files: {sorted(files)})"
    )
