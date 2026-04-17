# -*- coding: utf-8 -*-
"""
LLM 泛化引擎 —— 3 种可插拔后端。

用途：当 KG 的 L1/L2 都没命中或置信度低时，通过 LLM 在给定 KG 词表上下文里做泛化推理。

后端：
    1. EmbeddingEngine      — 用 Qwen3-Embedding-0.6B 做 zero-shot 分类（最快，已具备）
    2. LocalQwenEngine      — 本地加载 Qwen3-0.6B 生成模型（慢，但真正意义的 LLM）
    3. OpenAICompatEngine   — 调 OpenAI 兼容 API（DashScope / OpenAI / Moonshot / ...）

所有后端都实现统一接口：
    predict(title: str, context: str, kg_context_str: str, allowed_ids: set[str]) -> dict

输出：
    {
        "standard_id": "...",
        "standard_title": "...",
        "seniority": "",
        "domain": "",
        "confidence": 0.0,
        "reason": "",
        "raw": "..."         # 原始 LLM 输出（便于 debug）
    }
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from prompt_builder import (
    build_full_messages,
    extract_json_from_llm_output,
)

ROOT = Path(__file__).resolve().parent


class BaseLLMEngine(ABC):
    name: str = "base"

    @abstractmethod
    def predict(
        self,
        title: str,
        context: str,
        kg_context_str: str,
        allowed_ids: set[str],
        std_label_by_id: dict[str, str],
    ) -> dict: ...

    @staticmethod
    def _fallback_other(raw: str = "") -> dict:
        return {
            "standard_id": "other",
            "standard_title": "",
            "seniority": "",
            "domain": "",
            "confidence": 0.0,
            "reason": "llm_parse_failed",
            "raw": raw,
        }

    @staticmethod
    def _validate(result: dict, allowed_ids: set[str], std_label_by_id: dict[str, str]) -> dict:
        """约束 LLM 输出只能在 allowed_ids 范围内，否则降级为 other。"""
        sid = (result.get("standard_id") or "").strip()
        if sid and sid != "other" and sid not in allowed_ids:
            result["standard_id"] = "other"
            result["standard_title"] = ""
            result["reason"] = (result.get("reason", "") + " [forced_other:out_of_vocab]").strip()

        if result.get("standard_id") in allowed_ids:
            canonical = std_label_by_id.get(result["standard_id"], "")
            if canonical and not result.get("standard_title"):
                result["standard_title"] = canonical
        return result


# ══════════════════════════════════════════════════════════════════════
# Backend 1: Embedding-based zero-shot classification
# ══════════════════════════════════════════════════════════════════════

class EmbeddingEngine(BaseLLMEngine):
    """用 Qwen3-Embedding-0.6B 做最近邻分类 —— 把 embedding_service 的 search 结果格式化为 LLM 输出样式。"""

    name = "embedding"

    def __init__(
        self,
        embedding_kg=None,
        model_path: str | None = None,
        index_tag: str | None = None,
        top_k: int = 5,
        threshold: float = 0.45,
    ):
        from embedding_service import EmbeddingKG
        if embedding_kg is not None:
            self.ekg = embedding_kg
        else:
            kwargs = {}
            if model_path:
                kwargs["model_path"] = model_path
            if index_tag:
                kwargs["index_tag"] = index_tag
            self.ekg = EmbeddingKG(**kwargs)
        self.top_k = top_k
        self.threshold = threshold

    def _ensure_index(self) -> None:
        if self.ekg.embeddings is None:
            self.ekg.build_index()

    def predict(self, title, context, kg_context_str, allowed_ids, std_label_by_id) -> dict:
        self._ensure_index()
        query = title if not context else f"{title} | {context}"
        hits = self.ekg.search(query, top_k=self.top_k)
        if not hits:
            return self._fallback_other(raw="no_hits")

        top = hits[0]
        if top.score < self.threshold:
            res = {
                "standard_id": "other",
                "standard_title": "",
                "seniority": "",
                "domain": top.domain or "",
                "confidence": round(float(top.score), 4),
                "reason": f"低于阈值 {self.threshold}（top1={top.variant_name}）",
                "raw": str([h.to_dict() for h in hits]),
            }
        else:
            res = {
                "standard_id": top.standard_id,
                "standard_title": top.standard_label,
                "seniority": "",
                "domain": top.domain or "",
                "confidence": round(float(top.score), 4),
                "reason": f"embedding最近邻: {top.variant_name}",
                "raw": str([h.to_dict() for h in hits[:3]]),
            }
        return self._validate(res, allowed_ids, std_label_by_id)


# ══════════════════════════════════════════════════════════════════════
# Backend 2: Local Qwen3-0.6B generation model
# ══════════════════════════════════════════════════════════════════════

class LocalQwenEngine(BaseLLMEngine):
    """本地 transformers 加载 Qwen3-0.6B (chat/base) 做生成推理。"""

    name = "local_qwen"

    def __init__(
        self,
        model_path: str | None = None,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        device: str | None = None,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError("LocalQwenEngine 需要 transformers + torch") from e

        self.model_path = model_path or str(ROOT / "models" / "Qwen3-0.6B")
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"本地模型目录不存在: {self.model_path}\n"
                f"请先运行: python scripts/download_qwen_llm.py"
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LocalQwen] 加载生成模型: {self.model_path}  (device={self.device})")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
        ).to(self.device).eval()
        print(f"[LocalQwen] 加载完成 {time.time()-t0:.1f}s")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _generate(self, messages: list[dict]) -> str:
        import torch
        try:
            prompt_str = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            prompt_str = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

        inputs = self.tokenizer([prompt_str], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=(self.temperature > 0),
                temperature=max(self.temperature, 1e-5),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def predict(self, title, context, kg_context_str, allowed_ids, std_label_by_id) -> dict:
        messages = build_full_messages(title, context, kg_context_str)
        raw = self._generate(messages)
        parsed = extract_json_from_llm_output(raw)
        if not parsed:
            return {**self._fallback_other(raw=raw)}
        parsed["raw"] = raw[:400]
        return self._validate(parsed, allowed_ids, std_label_by_id)


# ══════════════════════════════════════════════════════════════════════
# Backend 3: OpenAI-compatible HTTP API
# ══════════════════════════════════════════════════════════════════════

class OpenAICompatEngine(BaseLLMEngine):
    """使用 OpenAI Chat Completions 兼容接口。

    支持：
        - OpenAI       BASE_URL=https://api.openai.com/v1                 MODEL=gpt-4o-mini
        - 阿里 DashScope BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1  MODEL=qwen3-0.6b
        - Moonshot     BASE_URL=https://api.moonshot.cn/v1                MODEL=moonshot-v1-8k
        - vLLM 自部署   BASE_URL=http://localhost:8000/v1                  MODEL=Qwen/Qwen3-0.6B

    环境变量：
        LLM_API_BASE     默认 https://dashscope.aliyuncs.com/compatible-mode/v1
        LLM_API_KEY      必填
        LLM_MODEL        默认 qwen3-0.6b
    """

    name = "openai_compat"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ):
        self.base_url = (base_url or os.environ.get("LLM_API_BASE")
                         or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or ""
        self.model = model or os.environ.get("LLM_MODEL") or "qwen3-0.6b"
        self.temperature = temperature
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError(
                "未设置 LLM_API_KEY 环境变量。\n"
                "PowerShell:  $env:LLM_API_KEY='sk-...'"
            )

    def _chat(self, messages: list[dict]) -> str:
        import httpx  # 轻量；若无则 fallback 到 requests
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        try:
            with httpx.Client(timeout=self.timeout) as cli:
                r = cli.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
        except Exception:
            payload.pop("response_format", None)
            with httpx.Client(timeout=self.timeout) as cli:
                r = cli.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
        return data["choices"][0]["message"]["content"]

    def predict(self, title, context, kg_context_str, allowed_ids, std_label_by_id) -> dict:
        messages = build_full_messages(title, context, kg_context_str)
        try:
            raw = self._chat(messages)
        except Exception as e:
            return {**self._fallback_other(raw=f"api_error: {e}")}
        parsed = extract_json_from_llm_output(raw)
        if not parsed:
            return {**self._fallback_other(raw=raw)}
        parsed["raw"] = raw[:400]
        return self._validate(parsed, allowed_ids, std_label_by_id)


# ══════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════

def build_engine(backend: str, **kwargs) -> BaseLLMEngine:
    backend = backend.lower()
    if backend in ("embedding", "emb"):
        return EmbeddingEngine(**kwargs)
    if backend in ("local", "local_qwen", "qwen"):
        return LocalQwenEngine(**kwargs)
    if backend in ("api", "openai", "dashscope", "openai_compat"):
        return OpenAICompatEngine(**kwargs)
    raise ValueError(f"unknown LLM backend: {backend}")


if __name__ == "__main__":
    import json as _json

    KG_PATH = ROOT / "knowledge_graph_v2.json"
    kg_raw = _json.loads(KG_PATH.read_text(encoding="utf-8"))
    allowed = {st["id"] for st in kg_raw["standard_titles"]}
    label_by_id = {st["id"]: st["label"] for st in kg_raw["standard_titles"]}

    from prompt_builder import build_kg_context
    kg_ctx = build_kg_context(kg_raw)

    engine = build_engine("embedding")
    for q in ["AI大模型产品负责人", "搞 K8s 的后端大哥", "做增长的 BD"]:
        r = engine.predict(q, "", kg_ctx, allowed, label_by_id)
        print(f"\n>> {q}")
        print(_json.dumps({k: r[k] for k in ["standard_id","standard_title","confidence","reason"]},
                          ensure_ascii=False, indent=2))
