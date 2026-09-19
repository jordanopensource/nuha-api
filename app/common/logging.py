"""Logging setup (text or JSON, env-selected).

Call ``setup_logging()`` once at process start (the classifier module's import
does); it is idempotent, so a second call replaces only its own handler.
"""

import logging
import os


_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def _resolve_level() -> str:
    raw = os.getenv("LOG_LEVEL", "INFO").upper()
    if raw not in _VALID_LOG_LEVELS:
        logging.getLogger(__name__).warning(
            "Invalid LOG_LEVEL %r, falling back to INFO. Valid: %s",
            raw,
            sorted(_VALID_LOG_LEVELS),
        )
        raw = "INFO"
    return raw


def setup_logging() -> None:
    """Configure root logging from LOG_LEVEL / LOG_FORMAT.

    Idempotent: a repeat call re-applies the level and replaces our own handler
    rather than stacking a second one (a combined test run, for instance, imports
    both services and so calls this twice in one process). Handlers installed by
    other code (pytest, uvicorn) are left untouched.
    """
    level_name = _resolve_level()
    level = getattr(logging, level_name, logging.INFO)
    log_format = os.getenv("LOG_FORMAT", "text").lower()

    if log_format == "json":
        import json as json_lib

        class JsonFormatter(logging.Formatter):
            def format(self, record):
                return json_lib.dumps(
                    {
                        "timestamp": self.formatTime(record),
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                    }
                )

        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    handler._nuha_handler = True  # tag our handler so a repeat call replaces it

    root = logging.getLogger()
    root.setLevel(level)
    for existing in [h for h in root.handlers if getattr(h, "_nuha_handler", False)]:
        root.removeHandler(existing)
    root.addHandler(handler)
