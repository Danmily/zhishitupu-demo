# -*- coding: utf-8 -*-
"""
下载 Qwen/Qwen3-Embedding-0.6B 模型到本地 `models/Qwen3-Embedding-0.6B` 目录。

优先使用 HuggingFace 官方镜像；如访问受限，可设置环境变量 HF_ENDPOINT=https://hf-mirror.com
再运行本脚本。

使用：
    python scripts/download_qwen_embedding.py

下载完成后可通过以下方式加载：
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("./models/Qwen3-Embedding-0.6B")
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
ROOT = Path(__file__).resolve().parent.parent
LOCAL_DIR = ROOT / "models" / "Qwen3-Embedding-0.6B"


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERR] 缺少依赖 huggingface_hub，请先执行：")
        print("      pip install -r requirements.txt")
        return 1

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"[INFO] HF_ENDPOINT = {endpoint}")
    print(f"[INFO] 下载模型: {MODEL_ID}")
    print(f"[INFO] 目标目录: {LOCAL_DIR}")
    print("-" * 60)

    try:
        path = snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(LOCAL_DIR),
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.md",
                "*.safetensors",
                "tokenizer*",
                "*.py",
            ],
        )
    except Exception as e:
        print(f"[ERR] 下载失败: {e}")
        print()
        print("如果是网络问题，可尝试使用镜像：")
        print("   PowerShell:  $env:HF_ENDPOINT='https://hf-mirror.com'; python scripts/download_qwen_embedding.py")
        print("   CMD:         set HF_ENDPOINT=https://hf-mirror.com && python scripts/download_qwen_embedding.py")
        return 2

    print("-" * 60)
    print(f"[DONE] 模型已下载至: {path}")

    files = sorted(Path(path).rglob("*"))
    total_bytes = sum(p.stat().st_size for p in files if p.is_file())
    print(f"[INFO] 文件数: {sum(1 for p in files if p.is_file())}, 总大小: {total_bytes/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
