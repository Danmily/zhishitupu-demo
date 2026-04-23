# -*- coding: utf-8 -*-
"""
混合 Pipeline —— KG 规则引擎 (L1/L2) + LLM 泛化引擎 (L3)。

调度逻辑（与领导达成的默认方案）：
    1. L1 精确匹配命中 → 直接返回（最便宜，零 LLM 调用）
    2. L2 模糊匹配，confidence >= threshold_l2 → 直接返回
    3. 以上都没解决 → 调用 LLM 在 KG 词表上下文里做泛化推理 → 返回
    4. LLM 返回 other / 非法 id → 最终 L0_miss

输出与旧 KG 保持兼容，多出 "path" 与 "trace" 两个字段：
    - path ∈ {"kg_l1", "kg_l2", "llm", "miss"}
    - trace: [每一步命中/跳过的简短说明]

用法：
    from pipeline import HybridPipeline
    from llm_engine import build_engine
    pl = HybridPipeline.from_defaults(llm_backend="embedding")
    print(pl.match("AI大模型产品负责人", context=""))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from app import TitleKnowledgeGraph
from llm_engine import BaseLLMEngine, EmbeddingEngine, build_engine
from prompt_builder import build_kg_context

ROOT = Path(__file__).resolve().parent
KG_PATH = ROOT / "knowledge_graph_v2.json"


class HybridPipeline:
    def __init__(
        self,
        kg: TitleKnowledgeGraph,
        kg_raw: dict,
        llm: Optional[BaseLLMEngine] = None,
        l2_confidence_threshold: float = 0.75,
    ):
        self.kg = kg
        self.kg_raw = kg_raw
        self.llm = llm
        self.l2_threshold = l2_confidence_threshold

        self.allowed_ids = {st["id"] for st in kg_raw["standard_titles"]}
        self.std_label_by_id = {st["id"]: st["label"] for st in kg_raw["standard_titles"]}
        self.std_category_by_id = {st["id"]: st["category"] for st in kg_raw["standard_titles"]}
        self._kg_context_cache: Optional[str] = None

    def _semantic_title_expansions(self, title: str, context: str = "", max_n: int = 20) -> list[dict]:
        """用 Embedding 索引对查询做近邻扩展，输出 10~20 条典型岗位写法（L3 / 检索泛化口径）。"""
        if not isinstance(self.llm, EmbeddingEngine):
            return []
        try:
            self.llm._ensure_index()
            q = f"{title} | {context}" if (context or "").strip() else title
            hits = self.llm.ekg.search(q, top_k=max(max_n * 2, 32))
        except Exception:
            return []
        out: list[dict] = []
        seen: set[str] = set()
        for h in hits:
            phrase = (h.variant_name or "").strip()
            if not phrase:
                continue
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "phrase": phrase,
                "standard_title": h.standard_label,
                "standard_id": h.standard_id,
                "similarity": round(float(h.score), 4),
            })
            if len(out) >= max_n:
                break
        return out

    def _attach_expansions(self, result: dict, title: str, context: str) -> dict:
        ex = self._semantic_title_expansions(title, context)
        if ex:
            result["semantic_expansions"] = ex
            result["semantic_expansions_note"] = (
                f"基于 Qwen3-Embedding 岗位词表近邻，共 {len(ex)} 条（用于展示/检索扩展，主结果仍由上层 L1/L2/L3 决策）"
            )
        else:
            result.setdefault("semantic_expansions", [])
        return result

    @classmethod
    def from_defaults(
        cls,
        llm_backend: Optional[str] = "embedding",
        llm_kwargs: Optional[dict] = None,
        l2_confidence_threshold: float = 0.75,
    ) -> "HybridPipeline":
        kg_raw = json.loads(KG_PATH.read_text(encoding="utf-8"))
        kg = TitleKnowledgeGraph(kg_raw)
        llm = build_engine(llm_backend, **(llm_kwargs or {})) if llm_backend else None
        return cls(kg, kg_raw, llm=llm, l2_confidence_threshold=l2_confidence_threshold)

    @property
    def kg_context(self) -> str:
        if self._kg_context_cache is None:
            self._kg_context_cache = build_kg_context(self.kg_raw)
        return self._kg_context_cache

    # ------------------------------------------------------------------ #
    # main entry
    # ------------------------------------------------------------------ #
    def match(self, title: str, context: str = "", *, include_semantic_expansions: bool = True) -> dict:
        trace: list[dict] = []

        r1 = self.kg.level1_exact(title)
        if r1:
            trace.append({"step": "L1_exact", "hit": True})
            out = self._pack(r1, path="kg_l1", trace=trace, level="L1_exact")
            return self._attach_expansions(out, title, context) if include_semantic_expansions else out
        trace.append({"step": "L1_exact", "hit": False})

        r2 = self.kg.level2_fuzzy(title)
        if r2 and (r2.get("confidence") or 0) >= self.l2_threshold:
            trace.append({"step": "L2_fuzzy",
                          "hit": True,
                          "confidence": r2["confidence"]})
            out = self._pack(r2, path="kg_l2", trace=trace, level="L2_fuzzy")
            return self._attach_expansions(out, title, context) if include_semantic_expansions else out
        trace.append({
            "step": "L2_fuzzy",
            "hit": bool(r2),
            "confidence": (r2.get("confidence") if r2 else 0),
            "threshold": self.l2_threshold,
            "passed": False,
        })

        if self.llm is None:
            trace.append({"step": "L3_llm", "skipped": "no_llm_configured"})
            miss = {
                "standard_id": None,
                "standard_title": None,
                "seniority": None,
                "domain": None,
                "level": "L0_miss",
                "path": "miss",
                "confidence": 0.0,
                "trace": trace,
            }
            return self._attach_expansions(miss, title, context) if include_semantic_expansions else miss

        llm_res = self.llm.predict(
            title=title,
            context=context,
            kg_context_str=self.kg_context,
            allowed_ids=self.allowed_ids,
            std_label_by_id=self.std_label_by_id,
        )
        trace.append({
            "step": "L3_llm",
            "backend": self.llm.name,
            "standard_id": llm_res.get("standard_id"),
            "confidence": llm_res.get("confidence"),
        })

        sid = llm_res.get("standard_id")
        if sid and sid in self.allowed_ids:
            ok = {
                "standard_id": sid,
                "standard_title": llm_res.get("standard_title")
                                  or self.std_label_by_id.get(sid, ""),
                "seniority": llm_res.get("seniority") or None,
                "domain": llm_res.get("domain") or None,
                "level": f"L3_llm_{self.llm.name}",
                "path": "llm",
                "confidence": float(llm_res.get("confidence") or 0.0),
                "reason": llm_res.get("reason", ""),
                "trace": trace,
            }
            return self._attach_expansions(ok, title, context) if include_semantic_expansions else ok

        bad = {
            "standard_id": None,
            "standard_title": None,
            "seniority": None,
            "domain": None,
            "level": "L0_miss",
            "path": "miss",
            "confidence": 0.0,
            "reason": llm_res.get("reason", "llm_returned_other"),
            "trace": trace,
        }
        return self._attach_expansions(bad, title, context) if include_semantic_expansions else bad

    # ------------------------------------------------------------------ #
    # util
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pack(r: dict, path: str, trace: list[dict], level: str) -> dict:
        return {
            "standard_id": r.get("standard_id"),
            "standard_title": r.get("standard_title"),
            "seniority": r.get("seniority"),
            "domain": r.get("domain"),
            "level": level,
            "path": path,
            "confidence": r.get("confidence"),
            "match_type": r.get("match_type"),
            "trace": trace,
        }


if __name__ == "__main__":
    pl = HybridPipeline.from_defaults(llm_backend="embedding")
    for q in [
        "产品经理",
        "高级产品经理",
        "AI大模型产品负责人",
        "搞 K8s 的后端大哥",
        "做增长的 BD",
        "咖啡师",
    ]:
        r = pl.match(q)
        print(f"\n>> {q}")
        print(f"   → standard_title={r['standard_title']}  path={r['path']}  "
              f"level={r['level']}  conf={r['confidence']}")
