# -*- coding: utf-8 -*-
"""
查询扩展（Query Expansion）模块。

输入：一个原始查询（如 "AI 开发工程师"）
输出：
    - 同义/相关岗位词集合（来自职位 KG）
    - 相关技能集合（来自 KG 的 related_skills + skill_graph 的 synonym/related）
    - ES 多字段 OR DSL（可直接给检索服务用，或本地 mock 演示）
    - 完整 trace（可视化用）

依赖：
    - app.TitleKnowledgeGraph —— 职位标准化 KG
    - skill_graph_v1.json     —— 技能标准化图
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent
SKILL_GRAPH_PATH = ROOT / "skill_graph_v1.json"


CATEGORY_HINTS: dict[str, list[str]] = {
    "AI/算法/大模型": ["AI", "人工智能", "算法", "大模型"],
    "后端开发": ["后端", "服务端", "backend"],
    "前端": ["前端", "web前端", "frontend"],
    "数据/分析": ["数据", "数据方向"],
    "产品": ["产品方向"],
    "运营/增长": ["运营", "增长"],
    "销售/商务": ["销售", "商务"],
    "HR/人力资源": ["人力资源", "人力"],
    "项目/管理": ["项目管理", "管理方向"],
    "财务/审计/法务": ["财务", "审计", "法务"],
    "测试/质量": ["测试", "QA"],
    "设计": ["设计方向"],
}


class SkillGraph:
    def __init__(self, path: Path = SKILL_GRAPH_PATH) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.version = data.get("version", "-")
        self.total = data.get("total_skills", 0)
        self.by_id: dict[str, dict] = {s["id"]: s for s in data["skills"]}
        self.by_label: dict[str, dict] = {}
        for s in data["skills"]:
            self.by_label[s["label"].lower()] = s
            for a in s.get("aliases", []):
                self.by_label.setdefault(a.lower(), s)
            for ch in (s.get("skill_tree") or {}).get("children") or []:
                for al in ch.get("aliases") or []:
                    key = (al or "").strip().lower()
                    if key:
                        self.by_label.setdefault(key, s)

    def lookup(self, name: str) -> Optional[dict]:
        return self.by_label.get((name or "").lower())

    def expand(self, seeds: list[str], max_hops: int = 2,
               top_related: int = 6) -> list[dict]:
        """从一组 seed 技能出发，扩展到 synonyms + related + skill_tree（有向子能力 + 无向组合边）。
        返回 [{"label", "via", "hops", "category"}, ...]，按 hops 升序、sim 降序。
        """
        visited: dict[str, dict] = {}
        for seed in seeds:
            s = self.lookup(seed)
            if not s:
                continue
            visited[s["id"]] = {
                "label": s["label"],
                "category": s["category"],
                "via": f"seed:{seed}",
                "hops": 0,
                "sim": 1.0,
            }

        frontier = list(visited.keys())
        for hop in range(1, max_hops + 1):
            next_frontier: list[str] = []
            for sid in frontier:
                s = self.by_id.get(sid)
                if not s:
                    continue
                neighbours = s.get("synonyms", []) + s.get("related", [])[:top_related]
                for nb in neighbours:
                    nid = nb["id"]
                    if nid in visited:
                        continue
                    ns = self.by_id.get(nid)
                    if not ns:
                        continue
                    visited[nid] = {
                        "label": ns["label"],
                        "category": ns["category"],
                        "via": f"{s['label']} -> {ns['label']} (sim={nb.get('sim','-')})",
                        "hops": hop,
                        "sim": nb.get("sim", 0),
                    }
                    next_frontier.append(nid)

                tree = s.get("skill_tree") or {}
                children = tree.get("children") or []
                child_by_id = {c["id"]: c for c in children if c.get("id")}
                activated_child_ids: set[str] = set()

                for ch in children:
                    ch_edge = ch.get("edge") or "directed"
                    ch_label = ch.get("label") or ""
                    added_from_ch = False
                    for tid in ch.get("ties") or []:
                        if tid in visited:
                            continue
                        ns = self.by_id.get(tid)
                        if not ns:
                            continue
                        visited[tid] = {
                            "label": ns["label"],
                            "category": ns["category"],
                            "via": f"{s['label']} ⊃ {ch_label} (tree/{ch_edge})",
                            "hops": hop,
                            "sim": 1.0,
                        }
                        next_frontier.append(tid)
                        added_from_ch = True
                    if added_from_ch and ch.get("id"):
                        activated_child_ids.add(ch["id"])

                for ce in tree.get("cross_edges") or []:
                    if (ce.get("edge") or "").lower() != "undirected":
                        continue
                    pair = ce.get("between") or []
                    if len(pair) < 2:
                        continue
                    a, b = pair[0], pair[1]
                    if not (child_by_id.get(a) and child_by_id.get(b)):
                        continue
                    note = ce.get("note") or "peer"
                    trigger = False
                    if a in activated_child_ids or b in activated_child_ids:
                        trigger = True
                    if trigger:
                        for cid in (a, b):
                            ch = child_by_id[cid]
                            if cid in activated_child_ids:
                                continue
                            for tid in ch.get("ties") or []:
                                if tid in visited:
                                    continue
                                ns = self.by_id.get(tid)
                                if not ns:
                                    continue
                                visited[tid] = {
                                    "label": ns["label"],
                                    "category": ns["category"],
                                    "via": f"{s['label']} · {note} (tree/undirected)",
                                    "hops": hop,
                                    "sim": 0.95,
                                }
                                next_frontier.append(tid)

            frontier = next_frontier
            if not frontier:
                break

        result = sorted(visited.values(), key=lambda x: (x["hops"], -float(x["sim"] or 0)))
        return result


def _collect_variants_for_std(kg, std: dict) -> list[str]:
    """把标准岗位的所有写法拉出来作为检索同义词"""
    out = [std["label"]]
    out += list(std.get("variants", []))
    out += list(std.get("senior_variants", []))
    for vs in std.get("domain_variants", {}).values():
        out += list(vs)
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        k = (x or "").strip()
        if not k or k.lower() in seen:
            continue
        seen.add(k.lower())
        uniq.append(k)
    return uniq


def build_es_dsl(job_variants: list[str], skill_labels: list[str],
                 title_field: str = "title",
                 skill_field: str = "skills",
                 desc_field: str = "description") -> dict:
    """生成一个 multi-field OR 的 ES DSL（bool/should）"""
    must_any: list[dict] = []
    if job_variants:
        must_any.append({
            "multi_match": {
                "query": " ".join(job_variants[:20]),
                "fields": [f"{title_field}^3", f"{desc_field}^1"],
                "type": "best_fields",
                "operator": "or",
            }
        })
    if skill_labels:
        must_any.append({
            "multi_match": {
                "query": " ".join(skill_labels[:30]),
                "fields": [f"{skill_field}^2", f"{desc_field}^1"],
                "type": "best_fields",
                "operator": "or",
            }
        })
    return {
        "query": {
            "bool": {
                "should": must_any,
                "minimum_should_match": 1,
            }
        },
        "size": 20,
        "_source": [title_field, skill_field, desc_field],
    }


def detect_skills_in_text(text: str, skill_graph: SkillGraph) -> list[dict]:
    """从 raw 文本里扫描出所有提到的技能（按 alias / label 精确匹配）。
    用于：当岗位 KG miss 时，从 query 直接抓技能当 seed。
    """
    hits: list[dict] = []
    seen: set[str] = set()
    text_norm = text or ""
    for label_lower, skill in skill_graph.by_label.items():
        if skill["id"] in seen:
            continue
        alias_list = [skill["label"]] + skill.get("aliases", [])
        for alias in alias_list:
            stripped = (alias or "").strip()
            if not stripped or len(stripped) < 2:
                continue
            if stripped in text_norm or stripped.lower() in text_norm.lower():
                hits.append({
                    "id": skill["id"],
                    "label": skill["label"],
                    "matched_alias": stripped,
                    "category": skill["category"],
                    "hit_type": "skill",
                })
                seen.add(skill["id"])
                break
    return hits


def detect_category_hints(text: str, skill_graph: SkillGraph,
                          top_per_category: int = 8) -> list[dict]:
    """如果 query 出现了大类提示词（"AI" / "人工智能" / "后端" / …），
    把对应大类下 doc_freq 最高的 top_per_category 个技能作为 "category_hint" seed。
    """
    out: list[dict] = []
    seen_categories: set[str] = set()
    for cat, hints in CATEGORY_HINTS.items():
        for h in hints:
            if h in text or h.lower() in text.lower():
                if cat in seen_categories:
                    break
                seen_categories.add(cat)
                skills_in_cat = [s for s in skill_graph.by_id.values() if s["category"] == cat]
                skills_in_cat.sort(key=lambda s: -int(s.get("doc_freq_asr", 0)))
                for s in skills_in_cat[:top_per_category]:
                    out.append({
                        "id": s["id"], "label": s["label"],
                        "matched_alias": h, "category": cat,
                        "hit_type": "category_hint",
                        "doc_freq_asr": s.get("doc_freq_asr", 0),
                    })
                break
    return out


def expand_query(
    raw_query: str,
    kg,
    kg_raw: dict,
    skill_graph: SkillGraph,
    include_skill_expansion: bool = True,
    max_skill_hops: int = 2,
    embedding_fallback_fn=None,
) -> dict:
    """主入口：输入 raw query，输出扩展结果 + ES DSL + trace。

    Args:
        embedding_fallback_fn: 可选的 callable(query) -> {"standard_id", "standard_title", "confidence"}
            当 KG L1/L2 都 miss 时，用它做 embedding 级兜底。
    """
    trace: list[dict] = []

    trace.append({"step": "normalize_query", "value_norm_query": kg._norm_query(raw_query)})
    r1 = kg.level1_exact(raw_query)
    r2 = None if r1 else kg.level2_fuzzy(raw_query)

    matched_std: list[dict] = []
    if r1:
        matched_std.append({"id": r1["standard_id"], "label": r1["standard_title"],
                            "match_level": r1["level"], "confidence": r1["confidence"]})
    elif r2 and (r2.get("confidence") or 0) >= 0.70:
        matched_std.append({"id": r2["standard_id"], "label": r2["standard_title"],
                            "match_level": r2["level"], "confidence": r2["confidence"]})

    if not matched_std and embedding_fallback_fn is not None:
        try:
            fb = embedding_fallback_fn(raw_query)
            if fb and fb.get("standard_id"):
                matched_std.append({
                    "id": fb["standard_id"],
                    "label": fb.get("standard_title", ""),
                    "match_level": "L3_embedding",
                    "confidence": fb.get("confidence", 0),
                })
        except Exception as e:
            trace.append({"step": "embedding_fallback_error", "error": str(e)})
    trace.append({"step": "match_job_kg", "hits": matched_std})

    skills_in_query = detect_skills_in_text(raw_query, skill_graph)
    trace.append({"step": "detect_skills_in_query", "hits": skills_in_query})

    category_hints = []
    if not matched_std and not skills_in_query:
        category_hints = detect_category_hints(raw_query, skill_graph)
        if category_hints:
            trace.append({"step": "detect_category_hints",
                          "hit_categories": sorted({h["category"] for h in category_hints}),
                          "expanded_seed_count": len(category_hints)})
            skills_in_query = skills_in_query + category_hints

    std_by_id = {st["id"]: st for st in kg_raw["standard_titles"]}
    job_variants: list[str] = []
    seed_skills: list[str] = []
    per_job_skills: list[dict] = []
    for m in matched_std:
        std = std_by_id.get(m["id"])
        if not std:
            continue
        variants = _collect_variants_for_std(kg, std)
        job_variants.extend(variants)
        rs = list(std.get("related_skills", []))
        seed_skills.extend(rs)
        per_job_skills.append({"job_id": std["id"], "job_label": std["label"],
                               "kg_related_skills": rs,
                               "variants_count": len(variants)})

    for s in skills_in_query:
        if s["label"] not in seed_skills:
            seed_skills.append(s["label"])

    trace.append({"step": "collect_job_variants_and_seed_skills",
                  "job_variants_count": len(job_variants),
                  "seed_skills_count": len(seed_skills),
                  "per_job": per_job_skills})

    job_variants = list(dict.fromkeys(job_variants))
    seed_skills = list(dict.fromkeys(seed_skills))

    expanded_skills: list[dict] = []
    if include_skill_expansion and seed_skills:
        expanded_skills = skill_graph.expand(seed_skills, max_hops=max_skill_hops)
        trace.append({"step": "expand_skills_via_skill_graph",
                      "hops": max_skill_hops, "expanded_count": len(expanded_skills)})

    all_skill_labels = [s["label"] for s in expanded_skills] or seed_skills
    es_dsl = build_es_dsl(job_variants, all_skill_labels)

    return {
        "input": raw_query,
        "matched_standard_jobs": matched_std,
        "skills_in_query": skills_in_query,
        "expanded_job_variants": job_variants,
        "seed_skills_from_kg": seed_skills,
        "expanded_skills": expanded_skills,
        "es_dsl": es_dsl,
        "trace": trace,
        "skill_graph_version": skill_graph.version,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    from app import TitleKnowledgeGraph

    kg_raw = json.loads((ROOT / "knowledge_graph_v2.json").read_text(encoding="utf-8"))
    kg = TitleKnowledgeGraph(kg_raw)
    sg = SkillGraph()

    demos = ["AI 开发工程师", "大模型应用工程师", "资深HRBP", "SaaS 大客户销售", "首席增长官"]
    for q in demos:
        print(f"\n{'='*72}\n>>> {q}")
        r = expand_query(q, kg, kg_raw, sg)
        print(f"  matched_std: {r['matched_standard_jobs']}")
        print(f"  job_variants ({len(r['expanded_job_variants'])}): "
              f"{r['expanded_job_variants'][:8]}...")
        print(f"  seed_skills ({len(r['seed_skills_from_kg'])}): "
              f"{r['seed_skills_from_kg'][:8]}")
        top_exp = [(s['label'], s['hops'], round(float(s['sim']),3))
                   for s in r['expanded_skills'][:10]]
        print(f"  expanded_skills top10 (label, hops, sim): {top_exp}")
