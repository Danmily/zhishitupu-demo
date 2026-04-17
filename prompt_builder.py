# -*- coding: utf-8 -*-
"""
为 LLM 泛化推理构建 prompt。

核心思路：
    - 把整张知识图谱（词表、类别、维度、交叉引用）"压缩"成一段紧凑的上下文
    - 本流程不做 RAG 动态召回，因为 19 个标准职称的 KG 非常小，可以一次性 feed
    - 强约束 JSON 输出，便于评测自动对齐
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
KG_PATH = ROOT / "knowledge_graph_v2.json"


SYSTEM_PROMPT = (
    "你是一名资深猎头/HR 领域的职位标准化专家，"
    "熟悉中国互联网/金融/制造等行业的岗位命名习惯、职级体系和领域细分。"
    "你的任务是把用户提到的职位描述，映射到下方给定的 19 个标准职称之一。"
    "必须严格按 JSON 输出，不要添加任何解释或 markdown 代码块。"
)


def build_kg_context(kg_raw: dict, max_variants_per_std: int = 10) -> str:
    lines: list[str] = []
    lines.append("# 职位标准化知识图谱")
    lines.append("")
    lines.append("## 标准职称（请从以下 standard_id 中选一个）")

    for st in kg_raw.get("standard_titles", []):
        variants = (st.get("variants", []) or []) + (st.get("senior_variants", []) or [])
        variants = variants[:max_variants_per_std]
        domains = list((st.get("domain_variants", {}) or {}).keys())
        skills = (st.get("related_skills", []) or [])[:6]

        parts = [
            f"- `{st['id']}` → **{st['label']}**  (类别: {st.get('category', '')})",
        ]
        if variants:
            parts.append(f"  同义/变体: {', '.join(variants)}")
        if domains:
            parts.append(f"  领域: {', '.join(domains)}")
        if skills:
            parts.append(f"  相关技能: {', '.join(skills)}")
        lines.append("\n".join(parts))

    lines.append("")
    lines.append("## 维度枚举")
    for key, dim in (kg_raw.get("dimensions", {}) or {}).items():
        values = ", ".join((dim.get("values") or [])[:12])
        lines.append(f"- **{key}** ({dim.get('label','')}): {values}")

    refs = kg_raw.get("cross_references") or []
    if refs:
        lines.append("")
        lines.append("## 相关参考（晋升路径 / 协作 / 同族）")
        for cr in refs[:10]:
            lines.append(
                f"- {cr.get('from','')} ←{cr.get('type','')}→ {cr.get('to','')}  "
                f"{cr.get('desc','')}"
            )

    return "\n".join(lines)


def build_user_prompt(title: str, context: str, kg_context: str) -> str:
    ctx_line = f"\n补充上下文: {context}" if context else ""
    return f"""{kg_context}

## 本次任务
用户输入职位描述:
  原始文本: {title}{ctx_line}

请思考：
  1. 这个职位最可能对应上面 19 个标准职称中的哪一个？
  2. 是否能识别出职级（seniority）？如"总监""VP""初级"等。
  3. 是否能识别出领域（domain）？如"AI""B端""金融"等。

严格输出 JSON（不要 markdown 代码块、不要任何额外文字），字段如下：
{{
  "standard_id": "上述 standard_id 之一，若完全无法匹配则写 other",
  "standard_title": "对应的中文 label，other 时留空字符串",
  "seniority": "从 seniority 枚举中挑一个，无则留空字符串",
  "domain": "从 domain 枚举中挑一个，无则留空字符串",
  "confidence": 0.0 到 1.0 的小数,
  "reason": "30 字以内说明关键判断依据"
}}
"""


def build_full_messages(title: str, context: str, kg_context: str) -> list[dict]:
    """OpenAI chat.completions 格式。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(title, context, kg_context)},
    ]


def extract_json_from_llm_output(text: str) -> dict[str, Any] | None:
    """容错地从 LLM 返回里抽取 JSON 对象。处理 markdown 围栏、前后废话等。"""
    if not text:
        return None
    s = text.strip()

    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3].strip()

    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    chunk = s[start : end + 1]

    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        chunk2 = chunk.replace("，", ",").replace("：", ":").replace("'", '"')
        try:
            return json.loads(chunk2)
        except json.JSONDecodeError:
            return None


def load_kg_context(kg_path: Path | str | None = None, **kwargs) -> str:
    """便捷函数：一行代码拿到 KG 上下文字符串。"""
    p = Path(kg_path) if kg_path else KG_PATH
    kg_raw = json.loads(p.read_text(encoding="utf-8"))
    return build_kg_context(kg_raw, **kwargs)


if __name__ == "__main__":
    ctx = load_kg_context()
    print(f"[KG context] {len(ctx)} chars, {ctx.count(chr(10))+1} lines")
    print("-" * 60)
    print(ctx[:800], "...")
    print("-" * 60)
    example = build_user_prompt("AI大模型产品负责人", "我们在做 LLM 应用", ctx)
    print(f"[User prompt] {len(example)} chars (total)")
    print(example[-600:])
