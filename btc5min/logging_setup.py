"""Terminal log capture — mirrors stdout/stderr to rotating files in logs/.

Writes:
  logs/btc5min.log  — full stream (INFO+)
  logs/errors.log   — stderr + tracebacks only

Kept minimal: Tee class duplicates writes; RotatingFileHandler caps size.
"""
from __future__ import annotations

import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOGS_DIR = _PROJECT_ROOT / "logs"


class _Tee:
    def __init__(self, original, file_handle, tag: str = ""):
        self._orig = original
        self._file = file_handle
        self._tag = tag

    def write(self, data):
        try:
            self._orig.write(data)
        except Exception:
            pass
        try:
            if data and data.strip():
                ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                prefix = f"[{ts}]" + (f"[{self._tag}] " if self._tag else " ")
                self._file.write(prefix + data if not data.startswith("\n") else data)
            else:
                self._file.write(data)
            self._file.flush()
        except Exception:
            pass

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass
        try:
            self._file.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self._orig.isatty()
        except Exception:
            return False

    def fileno(self):
        return self._orig.fileno()


def _open_rotating(path: Path, max_bytes: int = 10_000_000, backups: int = 5):
    """Open a rotating log file. Rotates when max_bytes exceeded."""
    handler = RotatingFileHandler(
        str(path), maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
    )
    # We only need the underlying stream; handler rotates on .emit().
    # Wrap its stream in a lightweight adapter that calls rotate via shouldRollover.
    return _HandlerStream(handler)


class _HandlerStream:
    def __init__(self, handler: RotatingFileHandler):
        self._h = handler

    def write(self, data: str):
        if not data:
            return
        # Fake a LogRecord just to trigger rotation logic with our raw text.
        stream = self._h.stream
        stream.write(data)
        # Manual rotation trigger based on file size.
        try:
            if self._h.maxBytes and stream.tell() >= self._h.maxBytes:
                self._h.doRollover()
        except Exception:
            pass

    def flush(self):
        try:
            self._h.flush()
        except Exception:
            pass


_installed = False


def install_terminal_logging(max_bytes: int = 10_000_000, backups: int = 5) -> Path:
    """Redirect stdout/stderr into logs/btc5min.log (+ errors.log for stderr).

    Idempotent. Returns the logs directory path.
    """
    global _installed
    if _installed:
        return _LOGS_DIR
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)

    main_stream = _open_rotating(_LOGS_DIR / "btc5min.log", max_bytes, backups)
    err_stream = _open_rotating(_LOGS_DIR / "errors.log", max_bytes, backups)

    sys.stdout = _Tee(sys.stdout, main_stream, tag="OUT")
    sys.stderr = _Tee(sys.stderr, err_stream, tag="ERR")
    # Also mirror stderr into the main log for unified timeline.
    sys.stderr = _Tee(sys.stderr, main_stream, tag="ERR")

    _installed = True
    banner = (
        f"\n{'='*60}\n"
        f"  TERMINAL LOGGING ACTIVO — {datetime.utcnow().isoformat()}Z\n"
        f"  Main:   {_LOGS_DIR / 'btc5min.log'}\n"
        f"  Errors: {_LOGS_DIR / 'errors.log'}\n"
        f"  Rotate: {max_bytes/1_000_000:.0f}MB x{backups} backups\n"
        f"{'='*60}\n"
    )
    print(banner)
    return _LOGS_DIR
