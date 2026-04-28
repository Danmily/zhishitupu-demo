# -*- coding: utf-8 -*-
"""兼容入口：技能子树已统一为 `ingest_excel_skill_trees.py`（多语言根节点）。"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ingest_excel_skill_trees import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
