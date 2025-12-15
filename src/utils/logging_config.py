from __future__ import annotations

import logging
from pathlib import Path
import os
import sys


def setup_logging(
    log_file: str = "logs/app.log",
    level: str | None = None,
) -> None:
    """
    ログ設定（UTF-8ファイル + コンソール）を行う。
    - Windows環境の文字化けを避けるため、ファイルはUTF-8で出力する
    - Streamlitの再実行でもハンドラが増殖しないようガードする
    """
    root = logging.getLogger()
    if getattr(root, "_configured_by_app", False):
        return

    lvl = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    try:
        log_level = getattr(logging, lvl)
    except Exception:
        log_level = logging.INFO

    root.setLevel(log_level)

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    sh = logging.StreamHandler(stream=sys.stderr)
    sh.setLevel(log_level)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # File (UTF-8)
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    root._configured_by_app = True


