# -*- coding: utf-8 -*-
"""
下载 Qwen3-0.6B 生成模型（用于 llm_engine.LocalQwenEngine）。

与 Qwen3-Embedding-0.6B 是两个不同的仓库：
    - Qwen/Qwen3-Embedding-0.6B   （已下载，用于向量检索）
    - Qwen/Qwen3-0.6B             （本脚本下载，用于 chat/generation）

使用：
    # 国内镜像（推荐）
    $env:HF_ENDPOINT='https://hf-mirror.com'
    python scripts/download_qwen_llm.py

    # 官方源
    python scripts/download_qwen_llm.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


MODEL_ID = "Qwen/Qwen3-0.6B"
ROOT = Path(__file__).resolve().parent.parent
LOCAL_DIR = ROOT / "models" / "Qwen3-0.6B"


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERR] 需先 pip install huggingface_hub")
        return 1

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"[INFO] HF_ENDPOINT = {endpoint}")
    print(f"[INFO] 模型: {MODEL_ID}  →  {LOCAL_DIR}")
    print("-" * 60)

    try:
        path = snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(LOCAL_DIR),
            allow_patterns=[
                "*.json", "*.txt", "*.md",
                "*.safetensors",
                "tokenizer*", "*.py",
            ],
        )
    except Exception as e:
        print(f"[ERR] 下载失败: {e}")
        print("可尝试镜像: $env:HF_ENDPOINT='https://hf-mirror.com'")
        return 2

    files = sorted(Path(path).rglob("*"))
    total = sum(p.stat().st_size for p in files if p.is_file())
    print("-" * 60)
    print(f"[DONE] 保存于 {path}")
    print(f"[INFO] 文件数 {sum(1 for p in files if p.is_file())}，总大小 {total/1e6:.1f} MB")
    print("\n加载测试:")
    print("  python -c \"from llm_engine import LocalQwenEngine; e=LocalQwenEngine(); print('ok')\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
