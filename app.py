# -*- coding: utf-8 -*-
"""
Knowledge Graph Job Title Standardization — Pure KG, Zero LLM
FastAPI backend: graph construction + multi-level matching + 80/20 generalization
"""

import json, re, sys, io, random
from pathlib import Path
from collections import defaultdict
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = Path(__file__).parent
KG_PATH = DATA_DIR / "knowledge_graph_v2.json"
TEST_PATH = DATA_DIR / "test_cases_v2.json"

KG_RAW = json.loads(KG_PATH.read_text(encoding="utf-8"))
TEST_CASES = json.loads(TEST_PATH.read_text(encoding="utf-8"))


# ═══════════════════════════════════════════════════════
#  Knowledge Graph Engine
# ═══════════════════════════════════════════════════════

class TitleKnowledgeGraph:

    def __init__(self, kg_data: dict):
        self.categories: list[str] = kg_data.get("categories", [])
        self.standards: dict[str, dict] = {}
        self.exact_map: dict[str, dict] = {}
        self.category_to_stds: dict[str, list] = defaultdict(list)
        self.std_to_variants: dict[str, list] = defaultdict(list)
        self.domain_keywords: set[str] = set()
        self.cross_refs = kg_data.get("cross_references", [])
        self.dimensions = kg_data.get("dimensions", {})

        self.seniority_patterns: list[tuple[str, str]] = [
            ("VP", "VP"), ("vp", "VP"),
            ("总监", "总监"), ("director", "总监"),
            ("主管", "主管"),
            ("负责人", "负责人"), ("leader", "负责人"), ("head", "负责人"),
            ("总经理", "高级"),
            ("高级", "高级/资深"), ("资深", "高级/资深"), ("senior", "高级/资深"),
            ("专家", "专家"), ("expert", "专家"),
            ("专员", "初级/助理"), ("助理", "初级/助理"), ("assistant", "初级/助理"),
            ("实习", "实习"), ("intern", "实习"),
        ]
        self._build(kg_data)

    # ── helpers ───────────────────────────────────────
    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"\s+", "", s).lower()

    def _detect_seniority_from_title(self, t: str) -> str | None:
        tl = t.lower()
        for kw in ("cpo", "coo", "cto", "cfo", "ceo", "cmo", "clo", "首席"):
            if kw in tl:
                return "C-level"
        if "vp" in tl:
            return "VP"
        if "总监" in tl or "director" in tl:
            return "总监"
        if "总经理" in tl:
            return "高级"
        return None

    # ── build graph ───────────────────────────────────
    def _build(self, kg: dict):
        for st in kg.get("standard_titles", []):
            sid, label, cat = st["id"], st["label"], st["category"]
            skills = st.get("related_skills", [])
            self.standards[sid] = {"label": label, "category": cat, "skills": skills}
            self.category_to_stds[cat].append(sid)

            self._add_edge(label, sid, label, "exact_standard", None, None)

            for v in st.get("variants", []):
                self._add_edge(v, sid, label, "variant", None, None)
                self.std_to_variants[sid].append({"name": v, "type": "variant", "seniority": None, "domain": None})

            for v in st.get("senior_variants", []):
                sen = self._detect_seniority_from_title(v)
                self._add_edge(v, sid, label, "senior_variant", sen, None)
                self.std_to_variants[sid].append({"name": v, "type": "senior_variant", "seniority": sen, "domain": None})

            for domain, vs in st.get("domain_variants", {}).items():
                self.domain_keywords.add(domain)
                for v in vs:
                    self._add_edge(v, sid, label, "domain_variant", None, domain)
                    self.std_to_variants[sid].append({"name": v, "type": "domain_variant", "seniority": None, "domain": domain})

    def _add_edge(self, variant, sid, slabel, mtype, seniority, domain):
        self.exact_map[self._norm(variant)] = {
            "standard_title": slabel,
            "standard_id": sid,
            "match_type": mtype,
            "seniority": seniority,
            "domain": domain,
            "original_variant": variant,
        }

    # ── node / edge counts ────────────────────────────
    def stats(self) -> dict:
        variant_types = defaultdict(int)
        for m in self.exact_map.values():
            variant_types[m["match_type"]] += 1
        return {
            "categories": len(self.categories),
            "standard_titles": len(self.standards),
            "total_edges": len(self.exact_map),
            "edges_by_type": dict(variant_types),
            "domain_keywords": sorted(self.domain_keywords),
            "cross_references": len(self.cross_refs),
        }

    # ── L1: exact lookup ──────────────────────────────
    def level1_exact(self, title: str) -> dict | None:
        r = self.exact_map.get(self._norm(title))
        if not r:
            return None
        return {
            **r,
            "level": "L1_exact",
            "confidence": 1.0,
            "trace": [
                {"step": "normalize", "value": self._norm(title)},
                {"step": "exact_lookup", "hit": True, "edge": f'{r["original_variant"]} --MAPS_TO--> {r["standard_title"]}'},
                {"step": "result", "standard_title": r["standard_title"], "match_type": r["match_type"]},
            ],
        }

    # ── L2: fuzzy matching ────────────────────────────
    def level2_fuzzy(self, title: str) -> dict | None:
        norm = self._norm(title)

        # 2a — strip common seniority/level prefixes
        prefixes = ["高级", "资深", "首席", "chief", "senior", "junior", "初级", "副"]
        for pfx in prefixes:
            stripped = norm.replace(pfx, "")
            if stripped != norm and stripped in self.exact_map:
                r = self.exact_map[stripped]
                return {
                    **r,
                    "level": "L2_fuzzy",
                    "match_type": "fuzzy_prefix_strip",
                    "confidence": 0.9,
                    "trace": [
                        {"step": "normalize", "value": norm},
                        {"step": "strip_prefix", "prefix": pfx, "result": stripped},
                        {"step": "exact_lookup", "hit": True, "edge": f'{r["original_variant"]} --MAPS_TO--> {r["standard_title"]}'},
                        {"step": "result", "standard_title": r["standard_title"]},
                    ],
                }

        # 2b — character Jaccard overlap
        best_score, best_key, best_r = 0.0, "", None
        for key, mapping in self.exact_map.items():
            s1, s2 = set(norm), set(key)
            union = len(s1 | s2)
            if union == 0:
                continue
            jaccard = len(s1 & s2) / union
            len_penalty = abs(len(norm) - len(key)) * 0.04
            score = jaccard - len_penalty
            if score > best_score:
                best_score, best_key, best_r = score, key, mapping

        if best_score >= 0.50 and best_r:
            return {
                **best_r,
                "level": "L2_fuzzy",
                "match_type": "fuzzy_char_overlap",
                "confidence": round(best_score, 3),
                "trace": [
                    {"step": "normalize", "value": norm},
                    {"step": "char_jaccard", "best_match": best_key, "jaccard": round(best_score, 3)},
                    {"step": "threshold_check", "threshold": 0.50, "passed": True},
                    {"step": "result", "standard_title": best_r["standard_title"]},
                ],
            }
        return None

    # ── L3: graph reasoning (token decomposition) ────
    def level3_graph_reasoning(self, title: str, context: str = "") -> dict | None:
        combined = title + " " + context
        scores: dict[str, float] = defaultdict(float)
        signals: dict[str, list] = defaultdict(list)
        inferred_seniority = None
        inferred_domain = None

        for pat, label in self.seniority_patterns:
            if pat in combined:
                inferred_seniority = label
                break

        for dkw in self.domain_keywords:
            if dkw in combined:
                inferred_domain = dkw
                break

        for sid, info in self.standards.items():
            label = info["label"]
            cat = info["category"]
            core = label.replace("经理", "").replace("工程师", "").replace("师", "").strip()

            if core and core in combined:
                scores[sid] += 10.0
                signals[sid].append(f'keyword("{core}") +10')
            if cat in combined:
                scores[sid] += 5.0
                signals[sid].append(f'category("{cat}") +5')

            if inferred_domain:
                for vinfo in self.std_to_variants.get(sid, []):
                    if vinfo.get("domain") == inferred_domain:
                        scores[sid] += 8.0
                        signals[sid].append(f'domain("{inferred_domain}") +8')
                        break

            for skill in info["skills"]:
                if skill in combined:
                    scores[sid] += 2.0
                    signals[sid].append(f'skill("{skill}") +2')

        if not scores:
            return None

        best_id = max(scores, key=scores.get)
        best_info = self.standards[best_id]
        top_signals = signals[best_id][:6]

        return {
            "standard_title": best_info["label"],
            "standard_id": best_id,
            "match_type": "graph_reasoning",
            "seniority": inferred_seniority,
            "domain": inferred_domain,
            "level": "L3_graph",
            "confidence": round(scores[best_id] / 20.0, 3),
            "trace": [
                {"step": "decompose", "tokens_found": top_signals},
                {"step": "score", "candidates": {sid: round(s, 1) for sid, s in sorted(scores.items(), key=lambda x: -x[1])[:5]}},
                {"step": "result", "standard_title": best_info["label"], "score": scores[best_id]},
            ],
        }

    # ── unified match ─────────────────────────────────
    def match(self, title: str, context: str = "") -> dict:
        for fn in [self.level1_exact, self.level2_fuzzy]:
            r = fn(title)
            if r:
                return r
        r = self.level3_graph_reasoning(title, context)
        if r:
            return r
        return {
            "standard_title": None, "standard_id": None,
            "match_type": None, "level": "L0_miss",
            "confidence": 0, "seniority": None, "domain": None,
            "trace": [{"step": "all_levels_exhausted", "result": "no_match"}],
        }


# ═══════════════════════════════════════════════════════
#  80/20 Generalization Experiment
# ═══════════════════════════════════════════════════════

def run_generalization(kg_raw: dict, rounds: int = 5, holdout_ratio: float = 0.2) -> dict:
    all_variants = []
    for st in kg_raw["standard_titles"]:
        sid, label = st["id"], st["label"]
        for v in st.get("variants", []):
            all_variants.append({"v": v, "sid": sid, "label": label})
        for v in st.get("senior_variants", []):
            all_variants.append({"v": v, "sid": sid, "label": label})
        for _d, vs in st.get("domain_variants", {}).items():
            for v in vs:
                all_variants.append({"v": v, "sid": sid, "label": label})

    results = []
    for rd in range(rounds):
        random.seed(42 + rd)
        shuffled = list(all_variants)
        random.shuffle(shuffled)
        cut = int(len(shuffled) * holdout_ratio)
        holdout, keep = shuffled[:cut], shuffled[cut:]
        keep_set = {item["v"] for item in keep}

        reduced = _reduce_kg(kg_raw, keep_set)
        g = TitleKnowledgeGraph(reduced)

        correct, details = 0, []
        for item in holdout:
            r = g.match(item["v"])
            ok = r["standard_title"] == item["label"]
            if ok:
                correct += 1
            details.append({"variant": item["v"], "expected": item["label"],
                            "got": r["standard_title"], "level": r["level"], "ok": ok})

        results.append({"round": rd + 1, "holdout": len(holdout), "correct": correct,
                         "accuracy": round(correct / len(holdout), 4) if holdout else 0,
                         "sample_failures": [d for d in details if not d["ok"]][:8]})

    avg = sum(r["accuracy"] for r in results) / len(results)
    return {"total_variants": len(all_variants), "holdout_ratio": holdout_ratio,
            "rounds": results, "avg_accuracy": round(avg, 4)}


def _reduce_kg(kg_raw: dict, keep: set) -> dict:
    out = {**kg_raw, "standard_titles": []}
    for st in kg_raw["standard_titles"]:
        nst = {**st,
               "variants": [v for v in st.get("variants", []) if v in keep],
               "senior_variants": [v for v in st.get("senior_variants", []) if v in keep],
               "domain_variants": {d: [v for v in vs if v in keep]
                                   for d, vs in st.get("domain_variants", {}).items()}}
        nst["domain_variants"] = {d: vs for d, vs in nst["domain_variants"].items() if vs}
        out["standard_titles"].append(nst)
    return out


# ═══════════════════════════════════════════════════════
#  Build Global Graph Instance
# ═══════════════════════════════════════════════════════

KG = TitleKnowledgeGraph(KG_RAW)
print(f"[KG] {KG.stats()['standard_titles']} standard titles, "
      f"{KG.stats()['total_edges']} edges, "
      f"{len(TEST_CASES)} test cases")


# ═══════════════════════════════════════════════════════
#  FastAPI
# ═══════════════════════════════════════════════════════

app = FastAPI(title="KG Pure Demo — Zero LLM")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class MatchReq(BaseModel):
    input_title: str
    context: str = ""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((Path(__file__).parent / "index.html").read_text(encoding="utf-8"))


@app.get("/api/stats")
async def stats():
    s = KG.stats()
    diff_c, cat_c = defaultdict(int), defaultdict(int)
    for t in TEST_CASES:
        diff_c[t["difficulty"]] += 1
        cat_c[t["category"]] += 1
    return {
        "kg": s,
        "tests_total": len(TEST_CASES),
        "tests_by_difficulty": dict(diff_c),
        "tests_by_category": dict(cat_c),
        "standard_titles": {sid: info["label"] for sid, info in KG.standards.items()},
    }


@app.get("/api/test_cases")
async def get_test_cases():
    return TEST_CASES


@app.post("/api/match")
async def match(req: MatchReq):
    r = KG.match(req.input_title, req.context)
    return r


@app.get("/api/batch_eval")
async def batch_eval():
    """Run all test cases through KG matching (instant, no LLM)."""
    results = []
    level_c = defaultdict(lambda: {"total": 0, "correct": 0})
    diff_c = defaultdict(lambda: {"total": 0, "correct": 0})
    cat_c = defaultdict(lambda: {"total": 0, "correct": 0})

    for tc in TEST_CASES:
        r = KG.match(tc["input_title"], tc.get("context", ""))
        ok = r.get("standard_title") == tc["expected_output"]["standard_title"]

        level_c[r["level"]]["total"] += 1
        diff_c[tc["difficulty"]]["total"] += 1
        cat_c[tc["category"]]["total"] += 1
        if ok:
            level_c[r["level"]]["correct"] += 1
            diff_c[tc["difficulty"]]["correct"] += 1
            cat_c[tc["category"]]["correct"] += 1

        results.append({
            "test_id": tc["test_id"],
            "input_title": tc["input_title"],
            "context": tc.get("context", ""),
            "difficulty": tc["difficulty"],
            "category": tc["category"],
            "target_level": tc.get("target_level", ""),
            "expected": tc["expected_output"]["standard_title"],
            "got": r.get("standard_title"),
            "level": r["level"],
            "match_type": r.get("match_type"),
            "confidence": r.get("confidence"),
            "ok": ok,
            "trace": r.get("trace", []),
        })

    total = len(results)
    correct = sum(1 for r in results if r["ok"])

    def _rate(d):
        return {k: {**v, "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0} for k, v in d.items()}

    return {
        "total": total, "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0,
        "by_level": _rate(level_c),
        "by_difficulty": _rate(diff_c),
        "by_category": _rate(cat_c),
        "details": results,
    }


@app.get("/api/generalization")
async def generalization():
    """80/20 generalization experiment (instant, no LLM)."""
    return run_generalization(KG_RAW)


@app.get("/api/graph_data")
async def graph_data():
    """Full graph structure for visualization."""
    nodes, edges = [], []

    for cat in KG.categories:
        nodes.append({"id": f"cat_{cat}", "label": cat, "type": "category", "size": 3})

    for sid, info in KG.standards.items():
        nodes.append({"id": sid, "label": info["label"], "type": "standard",
                       "category": info["category"], "size": 2,
                       "skills": info["skills"],
                       "variant_count": len(KG.std_to_variants.get(sid, []))})
        edges.append({"source": sid, "target": f"cat_{info['category']}", "type": "BELONGS_TO"})

    for sid, variants in KG.std_to_variants.items():
        for v in variants:
            vid = f"v_{KG._norm(v['name'])}"
            nodes.append({"id": vid, "label": v["name"], "type": v["type"],
                          "standard": KG.standards[sid]["label"], "size": 1,
                          "seniority": v.get("seniority"), "domain": v.get("domain")})
            edges.append({"source": vid, "target": sid, "type": "MAPS_TO"})

    for cr in KG.cross_refs:
        edges.append({"source": cr["from"], "target": cr["to"],
                       "type": cr["type"], "desc": cr.get("desc", "")})

    return {"nodes": nodes, "edges": edges,
            "summary": {"nodes": len(nodes), "edges": len(edges)}}


if __name__ == "__main__":
    import os, uvicorn
    port = int(os.environ.get("PORT", 8901))
    uvicorn.run(app, host="0.0.0.0", port=port)
