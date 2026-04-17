# -*- coding: utf-8 -*-
"""
基于 Qwen3-Embedding-0.6B 的本地向量检索服务。

职责：
    1. 加载本地 Qwen3-Embedding-0.6B 模型（无则自动从 HF 下载）
    2. 读取 dataset/kg_variants.csv，为每条变体（variant）离线构建 embedding 索引
    3. 提供 search(query, top_k) 接口：返回最相似的若干 variant 及对应标准职称
    4. 可作为 KG 的 L2.5 / L3+ 召回增强层

使用：
    from embedding_service import EmbeddingKG
    ekg = EmbeddingKG()
    ekg.build_index()            # 第一次需要（会缓存 .npy）
    print(ekg.search("AI大模型产品负责人", top_k=5))
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "Qwen3-Embedding-0.6B"
MODEL_ID_REMOTE = "Qwen/Qwen3-Embedding-0.6B"
VARIANTS_CSV = ROOT / "dataset" / "kg_variants.csv"
INDEX_ROOT = ROOT / "models" / "_kg_index"
INDEX_ROOT.mkdir(parents=True, exist_ok=True)

TASK_INSTRUCTION = (
    "Given a job title written by a user, retrieve the canonical or variant job "
    "title from the knowledge graph that most likely refers to the same role."
)


@dataclass
class SearchHit:
    variant_name: str
    standard_id: str
    standard_label: str
    variant_type: str
    domain: str
    score: float

    def to_dict(self) -> dict:
        return {
            "variant_name": self.variant_name,
            "standard_id": self.standard_id,
            "standard_label": self.standard_label,
            "variant_type": self.variant_type,
            "domain": self.domain,
            "score": round(float(self.score), 4),
        }


def _format_query(q: str) -> str:
    return f"Instruct: {TASK_INSTRUCTION}\nQuery: {q}"


class EmbeddingKG:
    def __init__(
        self,
        model_path: Optional[str] = None,
        index_tag: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_path: 模型目录。默认使用 ./models/Qwen3-Embedding-0.6B
            index_tag:  索引目录后缀，避免不同模型互相覆盖。
                        默认根据 model_path 自动生成（base / ft / <dir_name>）。
        """
        self.model_path = model_path or self._resolve_model_path()
        self.index_tag = index_tag or self._auto_tag(self.model_path)
        self.index_dir = INDEX_ROOT / self.index_tag
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.emb_file = self.index_dir / "variants_emb.npy"
        self.meta_file = self.index_dir / "variants_meta.csv"
        self.model = None
        self.embeddings: Optional[np.ndarray] = None
        self.meta: list[dict] = []

    @staticmethod
    def _resolve_model_path() -> str:
        if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
            return str(MODEL_DIR)
        return MODEL_ID_REMOTE

    @staticmethod
    def _auto_tag(model_path: str) -> str:
        p = Path(model_path)
        if not p.exists():
            return "remote"
        name = p.name
        return name.replace("/", "_").replace("\\", "_")

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "缺少 sentence-transformers，请先执行 pip install -r requirements.txt"
            ) from e

        print(f"[EmbeddingKG] 加载模型: {self.model_path}")
        self.model = SentenceTransformer(self.model_path, trust_remote_code=True)

    # ------------------------------------------------------------------ #
    # Index build / load
    # ------------------------------------------------------------------ #
    def _load_variants(self) -> list[dict]:
        rows: list[dict] = []
        with VARIANTS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        return rows

    def build_index(self, force: bool = False, batch_size: int = 32) -> None:
        if not force and self.emb_file.exists() and self.meta_file.exists():
            self.load_index()
            return

        self._ensure_model()
        variants = self._load_variants()
        corpus = [v["variant_name"] for v in variants]
        print(f"[EmbeddingKG] 构建索引 [{self.index_tag}]：{len(corpus)} 条变体")

        emb = self.model.encode(
            corpus,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.embeddings = emb.astype(np.float32)

        np.save(self.emb_file, self.embeddings)
        with self.meta_file.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "variant_id",
                    "variant_name",
                    "standard_id",
                    "standard_label",
                    "variant_type",
                    "domain",
                    "seniority_hint",
                ],
            )
            w.writeheader()
            for v in variants:
                w.writerow(v)
        self.meta = variants
        print(f"[EmbeddingKG] 索引已保存: {self.emb_file}, shape={self.embeddings.shape}")

    def load_index(self) -> None:
        if not self.emb_file.exists() or not self.meta_file.exists():
            raise FileNotFoundError(f"索引文件不存在 ({self.index_dir})，请先 build_index()")
        self.embeddings = np.load(self.emb_file)
        with self.meta_file.open("r", encoding="utf-8-sig", newline="") as f:
            self.meta = list(csv.DictReader(f))
        print(f"[EmbeddingKG] 加载索引 [{self.index_tag}]: shape={self.embeddings.shape}, meta={len(self.meta)}")

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        if self.embeddings is None or not self.meta:
            self.build_index()

        self._ensure_model()
        q_emb = self.model.encode(
            [_format_query(query)],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype(np.float32)

        scores = self.embeddings @ q_emb
        idx = np.argsort(-scores)[:top_k]

        hits: list[SearchHit] = []
        for i in idx:
            m = self.meta[int(i)]
            hits.append(
                SearchHit(
                    variant_name=m["variant_name"],
                    standard_id=m["standard_id"],
                    standard_label=m["standard_label"],
                    variant_type=m.get("variant_type", ""),
                    domain=m.get("domain", ""),
                    score=float(scores[int(i)]),
                )
            )
        return hits


if __name__ == "__main__":
    ekg = EmbeddingKG()
    ekg.build_index()
    for q in ["AI大模型产品负责人", "做推荐算法的哥们", "搞钱的VP", "搭K8s的"]:
        print("\n>>", q)
        for h in ekg.search(q, top_k=3):
            print(" ", h.to_dict())
