import logging


_DEF_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def init_logging(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format=_DEF_FORMAT)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger



class _ContextAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


class _DefaultContextFilter(logging.Filter):
    """Ensure third-party log records have run_id/stage fields expected by our formatter."""
    def __init__(self, run_id: str = "-", stage: str = "-") -> None:
        super().__init__()
        self._run_id = run_id or "-"
        self._stage = stage or "-"

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id"):
            setattr(record, "run_id", self._run_id)
        if not hasattr(record, "stage"):
            setattr(record, "stage", self._stage)
        return True


def _ensure_handler(level: int, fmt: str) -> None:
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(level)


def init_pipeline_logging(name: str,
                          run_id: str | None,
                          stage: str | None,
                          level: int = logging.INFO,
                          fmt: str = "logfmt") -> logging.Logger:
    """Initialize a logger with run_id and stage context.

    fmt: one of {"text", "logfmt", "json"}
    """
    if fmt == "json":
        # timestamp level logger name msg run_id stage
        pattern = "{" + ", ".join([
            "\"ts\": \"%(asctime)s\"",
            "\"level\": \"%(levelname)s\"",
            "\"logger\": \"%(name)s\"",
            "\"msg\": %(message)r",
            "\"run_id\": %(run_id)r",
            "\"stage\": %(stage)r",
        ]) + "}"
        fmt_str = pattern
    elif fmt == "logfmt":
        # ts=... level=... logger=... msg=... run_id=... stage=...
        fmt_str = "ts=%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s run_id=%(run_id)s stage=%(stage)s"
    else:
        fmt_str = _DEF_FORMAT + " - run_id=%(run_id)s - stage=%(stage)s"

    _ensure_handler(level, fmt_str)

    # Attach a default context filter to all handlers so third-party logs don't break formatting
    root = logging.getLogger()
    default_filter = _DefaultContextFilter(run_id or "-", stage or name)
    for h in root.handlers:
        # Avoid adding duplicate filters of the same type
        if not any(isinstance(f, _DefaultContextFilter) for f in h.filters):
            h.addFilter(default_filter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    ctx = {"run_id": run_id or "-", "stage": stage or name}
    return _ContextAdapter(logger, ctx)
