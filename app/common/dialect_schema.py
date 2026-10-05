"""Structural validation for dialect config files: the one schema every consumer imports.

A dialect is described by one JSON file (app/dialects/<code>.json in the repo,
installed next to its model as <models dir>/<code>/dialect.json). Everything
dialect-specific derives from it: name, the model source (``hf_repo``), languages
and aliases, preprocessing, labels, and how an ensemble's members are combined.

A dialect is always one ``hf_repo``. Whether that repo serves a single model or
an ensemble is a property of its snapshot, not of this file: a single-model
snapshot has one model.onnx at the top, an ensemble snapshot has a ``members/``
subdirectory with one complete model directory per member. The runtime tells the
two apart by that directory on the volume, and treats a single model as an
ensemble of one, so nothing downstream branches on the shape.

An optional ``aggregation`` block tunes how the members' logits become a
prediction. ``method`` names an entry in the engine's combine registry (a
default applies when omitted); ``bias`` is a per-sub-class vector added to the
combined logits before the argmax (this is where a dialect's bias lives, not in
a sidecar file in the snapshot). Like ``preprocessing.type``, whether a named
method actually exists is the engine's concern (its registry decides); the
schema only checks shape, plus the one cross-check it can make offline: a bias
must have exactly one entry per declared sub-class.

This module is the one definition of that schema; the runtime registry, the
fetch script, the pre-commit hook (via scripts/validate_dialects.py), and the
tests all import it.

Deliberately stdlib-only and network-free so the pre-commit hook stays fast and
offline. Out of scope on purpose (each stays where it belongs): whether
``preprocessing.type`` or ``aggregation.method`` name a real implementation
(that would couple the schema to the engine's registries and break the "adding
one is a code-only change" rule; the app checks it at load, the tests check it
for every file), and whether ``hf_repo`` actually resolves on HuggingFace (the
network model test and the fetch command cover that).
"""

import math

from app.common.config import MAX_LANG_LEN


# Dialect codes a dialect may not take: `api` is the service's compose name AND
# the dialect slot of its published tags (<channel>-api); `model` is the image
# grammar's component word from 1.x; `proxy` is reserved so a dialect can't take
# an infrastructure-sounding name. Rejected at validation time rather than
# producing a name collision in tags or on the models volume.
RESERVED_CODES = frozenset({"api", "model", "proxy"})

# The subdirectory inside a dialect directory that holds one complete model
# directory per ensemble member. Its presence on the volume is how the runtime
# and the fetch command tell an ensemble snapshot from a single-model one. Named
# here once so the engine, the registry, and the fetch command share the spelling.
MEMBERS_DIRNAME = "members"


def validate_dialect_config(code: str, cfg: dict) -> list[str]:
    """Return a list of structural problems with one dialect config (empty = OK).

    Mirrors what the app requires to load a dialect and what the test suite
    asserts, so a structurally broken (but valid-JSON) dialect file is rejected
    at commit time (pre-commit hook), at install time (fetch), and at load time
    (registry) with the same rules.
    """
    problems: list[str] = []

    def bad(msg: str) -> None:
        problems.append(f"{code}.json: {msg}")

    # The code is used verbatim as the {dialect} path segment the app routes on,
    # as the model's directory name on the models volume, and as an image-tag
    # component, so constrain it to ASCII lowercase letters; anything else breaks
    # routing or the directory/tag charset silently.
    if not (code.isascii() and code.islower() and code.isalpha()):
        bad("dialect code (the file stem) must be ASCII lowercase letters only (a-z)")
    # It shares the `lang` static cap: both are short routing tokens, and the
    # registry only considers directory names within the same bound.
    if len(code) > MAX_LANG_LEN:
        bad(f"dialect code (the file stem) must be at most {MAX_LANG_LEN} characters")
    if code in RESERVED_CODES:
        bad(
            f"dialect code {code!r} is reserved (would collide with the api "
            f"service name or the image tag grammar); rename the file"
        )

    missing = [
        f for f in ("name", "hf_repo", "languages", "preprocessing", "labels") if f not in cfg
    ]
    if missing:
        bad(f"missing required field(s): {', '.join(missing)}")
        return problems  # can't sensibly validate further without them

    if not (isinstance(cfg["name"], str) and cfg["name"].strip()):
        bad("'name' must be a non-empty string")

    problem = _check_repo(cfg["hf_repo"], "hf_repo")
    if problem:
        bad(problem)

    _validate_aggregation(cfg.get("aggregation"), cfg["labels"], bad)

    langs = cfg["languages"]
    if not (isinstance(langs, dict) and langs):
        bad("'languages' must be a non-empty object")
        langs = {}
    for lc, meta in langs.items():
        if not (
            isinstance(meta, dict) and isinstance(meta.get("name"), str) and meta["name"].strip()
        ):
            bad(f"language '{lc}' needs a non-empty 'name'")
        # A language code and its aliases are values a client may send as `lang`,
        # which the request schema caps at MAX_LANG_LEN before the value is even
        # seen. A declared code/alias longer than that could never match.
        if len(lc) > MAX_LANG_LEN:
            bad(f"language code '{lc}' must be at most {MAX_LANG_LEN} characters")
        aliases = meta.get("aliases", []) if isinstance(meta, dict) else None
        if aliases is not None and not (
            isinstance(aliases, list) and all(isinstance(a, str) and a.strip() for a in aliases)
        ):
            bad(f"language '{lc}' 'aliases' must be a list of non-empty strings")
        elif isinstance(aliases, list):
            for alias in aliases:
                if isinstance(alias, str) and len(alias) > MAX_LANG_LEN:
                    bad(
                        f"language '{lc}' alias '{alias}' must be at most {MAX_LANG_LEN} characters"
                    )

    # An alias is applied BEFORE the supported-language check at request time,
    # so an alias that equals a canonical code (or another alias) would
    # silently reroute that lang to a different language's labels.
    canonical_codes = set(langs)
    seen_aliases: dict[str, str] = {}
    for lc, meta in langs.items():
        aliases = meta.get("aliases", []) if isinstance(meta, dict) else []
        if not isinstance(aliases, list):
            continue
        for alias in aliases:
            if not isinstance(alias, str):
                continue
            if alias in canonical_codes:
                bad(f"language '{lc}' alias '{alias}' collides with a canonical language code")
            elif alias in seen_aliases:
                bad(f"alias '{alias}' is declared by both '{seen_aliases[alias]}' and '{lc}'")
            else:
                seen_aliases[alias] = lc

    prep = cfg["preprocessing"]
    if not (isinstance(prep, dict) and isinstance(prep.get("type"), str) and prep["type"].strip()):
        bad("'preprocessing' must be an object with a non-empty 'type'")

    problems.extend(_validate_labels(code, cfg["labels"], set(langs)))
    return problems


def _check_repo(repo: object, label: str) -> str | None:
    """The 'namespace/name' grammar an hf_repo id must follow."""
    if not (isinstance(repo, str) and repo.strip()):
        return f"'{label}' must be a non-empty string"
    if (
        "://" in repo
        or any(c.isspace() for c in repo)
        or [p for p in repo.split("/") if p] != repo.split("/")
        or repo.count("/") != 1
    ):
        return f"'{label}' must be a 'namespace/name' id, got {repo!r}"
    return None


def _validate_aggregation(agg: object, labels: object, bad) -> None:
    """The optional aggregation block: a combine method name (shape only, the
    engine owns the registry) and a per-sub-class bias vector (shape plus the one
    offline cross-check, that it has one entry per declared sub-class)."""
    if agg is None:
        return
    if not isinstance(agg, dict):
        bad("'aggregation' must be an object")
        return
    if "method" in agg and not (isinstance(agg["method"], str) and agg["method"].strip()):
        bad("'aggregation.method' must be a non-empty string naming a combine method")
    if "bias" in agg:
        bias = agg["bias"]
        expected = _sub_class_count(labels)
        if not (isinstance(bias, list) and bias):
            bad("'aggregation.bias' must be a non-empty list of numbers")
        elif any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            for x in bias
        ):
            # json.load accepts NaN/Infinity; a non-finite bias would poison the
            # softmax into NaN at serve time, so reject it here at commit/install.
            bad("'aggregation.bias' must be a list of finite numbers")
        elif expected is not None and len(bias) != expected:
            bad(
                f"'aggregation.bias' has {len(bias)} entries but the labels declare "
                f"{expected} sub-classes"
            )


def _sub_class_count(labels: object) -> int | None:
    """The number of sub-classes the labels declare (one model logit each), or
    None when the labels are too malformed to tell (the labels validator reports
    that separately)."""
    if (
        isinstance(labels, dict)
        and isinstance(labels.get("sub_to_main"), dict)
        and labels["sub_to_main"]
    ):
        return len(labels["sub_to_main"])
    return None


def _validate_labels(code: str, labels: object, declared_langs: set[str]) -> list[str]:
    """Validate the labels block: sub/main/sub_to_main present, label languages
    match the declared languages, digit keys, non-empty strings, consistent key
    sets across languages, and a sub_to_main that maps every sub id onto an
    existing main id (no orphans, no dangling targets)."""
    problems: list[str] = []

    def bad(msg: str) -> None:
        problems.append(f"{code}.json: {msg}")

    if not isinstance(labels, dict):
        bad("'labels' must be an object")
        return problems
    for key in ("sub", "main", "sub_to_main"):
        if not (isinstance(labels.get(key), dict) and labels[key]):
            bad(f"labels.{key} must be a non-empty object")
    if problems:
        return problems  # too broken to cross-check

    for cat in ("sub", "main"):
        if set(labels[cat]) != declared_langs:
            bad(
                f"labels.{cat} languages {sorted(labels[cat])} != "
                f"declared languages {sorted(declared_langs)}"
            )
        reference_keys = None
        for lc, mapping in labels[cat].items():
            if not (isinstance(mapping, dict) and mapping):
                bad(f"labels.{cat}.{lc} must be a non-empty object")
                continue
            for label_key, value in mapping.items():
                if not (isinstance(label_key, str) and label_key.isdigit()):
                    bad(f"labels.{cat}.{lc} key {label_key!r} must be a digit string")
                if not (isinstance(value, str) and value.strip()):
                    bad(f"labels.{cat}.{lc}[{label_key}] must be a non-empty string")
            if reference_keys is None:
                reference_keys = set(mapping)
            elif set(mapping) != reference_keys:
                bad(f"labels.{cat}.{lc} keys differ from the other languages")

    s2m = labels["sub_to_main"]
    sub_ids = {int(k) for k in next(iter(labels["sub"].values())) if str(k).isdigit()}
    main_ids = {int(k) for k in next(iter(labels["main"].values())) if str(k).isdigit()}
    for key, target in s2m.items():
        if not (isinstance(key, str) and key.isdigit()):
            bad(f"sub_to_main key {key!r} must be a digit string")
        if isinstance(target, bool) or not isinstance(target, int):
            bad(f"sub_to_main[{key}] must be an integer main id")
        elif target not in main_ids:
            bad(f"sub_to_main[{key}] -> {target} has no matching main label")
    mapped = {int(k) for k in s2m if str(k).isdigit()}
    if sub_ids - mapped:
        bad(f"sub ids {sorted(sub_ids - mapped)} have no sub_to_main mapping")
    if mapped - sub_ids:
        bad(f"sub_to_main has keys {sorted(mapped - sub_ids)} not present in sub labels")
    return problems
