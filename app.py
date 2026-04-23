# -*- coding: utf-8 -*-
"""
Knowledge Graph Job Title Standardization — Pure KG, Zero LLM
FastAPI backend: graph construction + multi-level matching + 80/20 generalization
"""

import json, re, sys, io, random, csv, threading, time
from pathlib import Path
from collections import defaultdict
from fastapi import FastAPI, HTTPException
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

    _QUERY_STRIP_PREFIXES = (
        "首席", "资深", "高级", "初级", "副",
        "chief", "senior", "junior", "sr.", "jr.",
    )

    @classmethod
    def _norm_query(cls, s: str) -> str:
        """查询侧专用 normalize：

        相较 ``_norm`` 多做两件事，用来吸收真实猎头场景里常见的噪声修饰：

        1. 剥离全角/半角圆括号内内容，例如 ``X（偏策略）`` → ``X``
        2. 反复剥离『首席/资深/高级/初级/副』等职级前缀，支持组合叠加
           例如 ``首席高级Java工程师`` → ``Java工程师``

        仅用于查询，不会改 KG 构建时的索引 key，因此不会污染元信息。
        """
        s = re.sub(r"[（(][^）)]*[）)]", "", s)
        s = re.sub(r"\s+", "", s).lower()
        changed = True
        while changed:
            changed = False
            for p in cls._QUERY_STRIP_PREFIXES:
                if s.startswith(p) and len(s) > len(p):
                    s = s[len(p):]
                    changed = True
                    break
        return s

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
        norm = self._norm(title)
        r = self.exact_map.get(norm)
        if r:
            return {
                **r,
                "level": "L1_exact",
                "confidence": 1.0,
                "trace": [
                    {"step": "normalize", "value": norm},
                    {"step": "exact_lookup", "hit": True, "edge": f'{r["original_variant"]} --MAPS_TO--> {r["standard_title"]}'},
                    {"step": "result", "standard_title": r["standard_title"], "match_type": r["match_type"]},
                ],
            }

        deep = self._norm_query(title)
        if deep and deep != norm:
            r = self.exact_map.get(deep)
            if r:
                return {
                    **r,
                    "level": "L1_exact",
                    "match_type": r["match_type"] + "_after_strip",
                    "confidence": 0.95,
                    "trace": [
                        {"step": "normalize", "value": norm},
                        {"step": "strip_modifiers", "deep_norm": deep,
                         "hint": "剥离括号注释与『首席/资深/高级/初级』等前缀"},
                        {"step": "exact_lookup", "hit": True,
                         "edge": f'{r["original_variant"]} --MAPS_TO--> {r["standard_title"]}'},
                        {"step": "result", "standard_title": r["standard_title"],
                         "match_type": r["match_type"] + "_after_strip"},
                    ],
                }
        return None

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


GENERALIZATION_HYBRID_CACHE = DATA_DIR / "dataset" / "generalization_8020_hybrid.json"


@app.get("/api/generalization")
async def generalization():
    """80/20 泛化：默认即时跑纯 KG；若存在离线报表则附带 KG+Embedding(L3) 对比。"""
    base = run_generalization(KG_RAW)
    base["kg_plus_embedding"] = None
    if GENERALIZATION_HYBRID_CACHE.exists():
        try:
            base["kg_plus_embedding"] = json.loads(
                GENERALIZATION_HYBRID_CACHE.read_text(encoding="utf-8")
            )
        except Exception:
            pass
    return base


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


BUSINESS_DATA_DIR = DATA_DIR / "dataset" / "business"
BUSINESS_MANIFEST_PATH = BUSINESS_DATA_DIR / "business_manifest.json"


@app.get("/api/business_data/summary")
async def business_data_summary():
    """研发业务导出清单：需先运行 scripts/ingest_business_excel.py 生成 manifest。"""
    if not BUSINESS_MANIFEST_PATH.exists():
        return {
            "loaded": False,
            "hint": "python scripts/ingest_business_excel.py --skills <技能.xlsx> --titles <职称.xlsx>",
            "spec": "openspec/specs/business-talent-exports/spec.md",
        }
    data = json.loads(BUSINESS_MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"loaded": True, **data}


@app.get("/api/business_data/title_freq")
async def business_data_title_freq(limit: int = 100):
    path = BUSINESS_DATA_DIR / "talent_titles_top_freq.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="未找到 talent_titles_top_freq.csv，请先导入业务数据")
    limit = max(1, min(int(limit), 8000))
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= limit:
                break
            rows.append(row)
    return {"limit": limit, "rows": rows}


@app.get("/api/business_data/skill_key_summary")
async def business_data_skill_key_summary():
    path = BUSINESS_DATA_DIR / "talent_skills_by_key_summary.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="未找到 talent_skills_by_key_summary.csv，请先导入业务数据")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {"rows": rows}


# ═══════════════════════════════════════════════════════
#  Hybrid Pipeline (KG + Embedding LLM) —— 懒加载 + 后台预热
# ═══════════════════════════════════════════════════════
#
# 这段代码把我们新开发的 pipeline.py（L1/L2 规则 + L3 Embedding 泛化）
# 接入到 FastAPI 中。模型加载比较重（0.6B Embedding + 629 变体向量索引），
# 所以采取"启动时后台线程预热 + 请求时若未就绪则 503 + 前端轮询 status"
# 的方式，保证纯 KG 接口零延迟启动、混合链路按需就绪。

DATASET_DIR = DATA_DIR / "dataset"

_PIPELINE = None
_PIPELINE_READY = False
_PIPELINE_LOADING = False
_PIPELINE_ERROR: str | None = None
_PIPELINE_BACKEND = None  # 实际加载的模型类型（base / ft）
_PIPELINE_LOCK = threading.Lock()


def _pick_embedding_model_path() -> tuple[str | None, str]:
    """优先选择已微调模型，否则回退基础模型。返回 (path_or_None, tag)。"""
    ft = DATA_DIR / "models" / "Qwen3-Embedding-0.6B-ft"
    base = DATA_DIR / "models" / "Qwen3-Embedding-0.6B"
    if (ft / "config.json").exists():
        return str(ft), "ft"
    if (base / "config.json").exists():
        return str(base), "base"
    return None, "missing"


def _load_pipeline_background():
    """在后台线程中加载 HybridPipeline，避免阻塞 FastAPI 启动。"""
    global _PIPELINE, _PIPELINE_READY, _PIPELINE_LOADING, _PIPELINE_ERROR, _PIPELINE_BACKEND
    with _PIPELINE_LOCK:
        if _PIPELINE_READY or _PIPELINE_LOADING:
            return
        _PIPELINE_LOADING = True
    t0 = time.time()
    try:
        from pipeline import HybridPipeline
        model_path, tag = _pick_embedding_model_path()
        if model_path is None:
            raise RuntimeError(
                "未检测到本地 Qwen3-Embedding-0.6B 模型，请先运行 "
                "`python scripts/download_qwen_embedding.py`"
            )
        kwargs = {"model_path": model_path, "index_tag": tag}
        _PIPELINE = HybridPipeline.from_defaults(
            llm_backend="embedding", llm_kwargs=kwargs,
        )
        _PIPELINE_BACKEND = tag
        _PIPELINE_READY = True
        print(f"[Pipeline] ready in {time.time()-t0:.1f}s, backend={tag}, path={model_path}")
    except Exception as e:
        _PIPELINE_ERROR = f"{type(e).__name__}: {e}"
        print(f"[Pipeline] load failed: {_PIPELINE_ERROR}")
    finally:
        _PIPELINE_LOADING = False


@app.on_event("startup")
def _startup_warmup():
    """服务一起来就后台加载 pipeline，前端立刻可用纯 KG 接口。"""
    threading.Thread(target=_load_pipeline_background, daemon=True).start()


@app.get("/api/pipeline_status")
async def pipeline_status():
    model_path, tag = _pick_embedding_model_path()
    return {
        "ready": _PIPELINE_READY,
        "loading": _PIPELINE_LOADING,
        "error": _PIPELINE_ERROR,
        "backend": _PIPELINE_BACKEND,
        "embedding_model_tag": tag,
        "embedding_model_path": model_path,
        "fine_tuned_available": tag == "ft",
    }


@app.post("/api/pipeline_warmup")
async def pipeline_warmup():
    """手动触发一次预热（通常不需要，@startup 已经触发过了）。"""
    if _PIPELINE_READY:
        return {"ok": True, "status": "already_ready"}
    threading.Thread(target=_load_pipeline_background, daemon=True).start()
    return {"ok": True, "status": "loading_started"}


@app.post("/api/pipeline_match")
async def pipeline_match(req: MatchReq):
    """单条走 KG + Embedding 混合链路，返回完整 trace。"""
    if not _PIPELINE_READY:
        raise HTTPException(
            status_code=503,
            detail={
                "loading": _PIPELINE_LOADING,
                "error": _PIPELINE_ERROR,
                "message": "Embedding 模型仍在加载中，请稍后重试（通常 30-60 秒）",
            },
        )
    r = _PIPELINE.match(req.input_title, req.context)
    return r


def _csv_to_list(path: Path) -> list[dict]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"CSV 文件不存在：{path.name}")
    with path.open("r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


@app.get("/api/pipeline_report")
async def pipeline_report():
    """
    直接返回离线跑好的 300 条混合链路评测结果（dataset/pipeline_eval_report.csv）。
    汇报页面秒开用。包含 KG-only 与 KG+LLM 两条链路的并排对比。
    """
    rows = _csv_to_list(DATASET_DIR / "pipeline_eval_report.csv")

    total = len(rows)
    kg_correct = sum(1 for r in rows if (r.get("kg_got") or "") == r.get("expected", ""))
    final_correct = sum(1 for r in rows if r.get("final_correct") == "1")

    source_counter: dict[str, int] = defaultdict(int)
    llm_rescue = 0   # KG 没解决 + LLM 解对了
    llm_wrong = 0    # KG 没解决 + LLM 解错了
    kg_only_missed_llm_saved_cases = []

    for r in rows:
        source_counter[r.get("final_source", "") or "unknown"] += 1
        kg_ok = (r.get("kg_got") or "") == r.get("expected", "")
        final_ok = r.get("final_correct") == "1"
        if not kg_ok and r.get("final_source") == "llm":
            if final_ok:
                llm_rescue += 1
                if len(kg_only_missed_llm_saved_cases) < 30:
                    kg_only_missed_llm_saved_cases.append({
                        "test_id": r["test_id"],
                        "input_title": r["input_title"],
                        "context": r.get("context", ""),
                        "difficulty": r.get("difficulty", ""),
                        "expected": r.get("expected", ""),
                        "kg_got": r.get("kg_got", ""),
                        "kg_level": r.get("kg_level", ""),
                        "final_got": r.get("final_got", ""),
                    })
            else:
                llm_wrong += 1

    # 按目标层级拆分两条链路准确率
    by_target: dict[str, dict] = defaultdict(lambda: {"total": 0, "kg_correct": 0, "final_correct": 0})
    for r in rows:
        k = r.get("target_level") or "-"
        by_target[k]["total"] += 1
        if (r.get("kg_got") or "") == r.get("expected", ""):
            by_target[k]["kg_correct"] += 1
        if r.get("final_correct") == "1":
            by_target[k]["final_correct"] += 1
    for v in by_target.values():
        v["kg_acc"] = round(v["kg_correct"] / v["total"], 4) if v["total"] else 0
        v["final_acc"] = round(v["final_correct"] / v["total"], 4) if v["total"] else 0

    # 补充“1500 全量口径”：
    # - 若当前报表本身就是 1500 条，则 directly measured
    # - 若当前仅 300 条，则按 test_cases_v2 的 target_level 分布做加权推算
    full_target_counter: dict[str, int] = defaultdict(int)
    for tc in TEST_CASES:
        full_target_counter[tc.get("target_level") or "-"] += 1

    full_total = len(TEST_CASES)
    if total == full_total:
        full_view = {
            "mode": "measured",
            "total": full_total,
            "kg_only": {
                "correct": kg_correct,
                "accuracy": round(kg_correct / total, 4) if total else 0,
            },
            "hybrid": {
                "correct": final_correct,
                "accuracy": round(final_correct / total, 4) if total else 0,
            },
            "uplift_points": round((final_correct - kg_correct) / total, 4) if total else 0,
            "note": "当前报表已是 1500 条全量实测结果。",
        }
    else:
        est_kg_correct = 0.0
        est_final_correct = 0.0
        for k, n in full_target_counter.items():
            # 当前报表没覆盖到的 target_level，按 0 处理，避免误报
            m = by_target.get(k, {"kg_acc": 0.0, "final_acc": 0.0})
            est_kg_correct += n * float(m.get("kg_acc", 0.0))
            est_final_correct += n * float(m.get("final_acc", 0.0))
        full_view = {
            "mode": "estimated_from_current_report",
            "total": full_total,
            "kg_only": {
                "correct": round(est_kg_correct),
                "accuracy": round(est_kg_correct / full_total, 4) if full_total else 0,
            },
            "hybrid": {
                "correct": round(est_final_correct),
                "accuracy": round(est_final_correct / full_total, 4) if full_total else 0,
            },
            "uplift_points": round((est_final_correct - est_kg_correct) / full_total, 4) if full_total else 0,
            "note": (
                f"当前报表仅 {total} 条，以上为按 test_cases_v2.json 全量分布（1500 条）"
                "加权推算，用于汇报口径；最终以 1500 条实测为准。"
            ),
        }

    return {
        "total": total,
        "kg_only": {
            "correct": kg_correct,
            "accuracy": round(kg_correct / total, 4) if total else 0,
        },
        "hybrid": {
            "correct": final_correct,
            "accuracy": round(final_correct / total, 4) if total else 0,
        },
        "uplift_points": round((final_correct - kg_correct) / total, 4) if total else 0,
        "sources": dict(source_counter),
        "llm_rescue_count": llm_rescue,
        "llm_wrong_count": llm_wrong,
        "by_target_level": dict(by_target),
        "full_1500_view": full_view,
        "llm_rescue_samples": kg_only_missed_llm_saved_cases,
        "details": rows,
    }


@app.get("/api/llm_failures")
async def llm_failures():
    """失败反例聚合（按期望标准职称分组）——词表优化的输入。"""
    rows = _csv_to_list(DATASET_DIR / "llm_failures_by_std.csv")
    for r in rows:
        if "fail_count" in r:
            try:
                r["fail_count"] = int(r["fail_count"])
            except Exception:
                pass
    return {"total_groups": len(rows), "rows": rows}


@app.get("/api/kg_expand_candidates")
async def kg_expand_candidates():
    """LLM 驱动的词表扩充候选（基于失败反例自动生成）。"""
    rows = _csv_to_list(DATASET_DIR / "kg_expand_candidates.csv")
    return {"total": len(rows), "rows": rows}


# ═══════════════════════════════════════════════════════
#  Skill Graph + Query Expansion (任务 2~4)
# ═══════════════════════════════════════════════════════

try:
    from query_expansion import SkillGraph, expand_query as _do_expand_query
    _SKILL_GRAPH: SkillGraph | None = SkillGraph()
    print(f"[SkillGraph] loaded v{_SKILL_GRAPH.version}, {_SKILL_GRAPH.total} skills")
except Exception as e:
    _SKILL_GRAPH = None
    print(f"[SkillGraph] load failed: {e}")


class ExpandQueryReq(BaseModel):
    query: str
    include_skill_expansion: bool = True
    max_skill_hops: int = 2
    use_embedding_fallback: bool = True


@app.get("/api/skill_graph_info")
async def skill_graph_info():
    if _SKILL_GRAPH is None:
        raise HTTPException(status_code=503, detail="skill_graph_v1.json 未加载成功")
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for s in _SKILL_GRAPH.by_id.values():
        row = {
            "label": s["label"], "aliases": s.get("aliases", []),
            "doc_freq_asr": s.get("doc_freq_asr", 0),
            "in_kg_jobs": s.get("in_kg_jobs", []),
            "n_synonyms": len(s.get("synonyms", [])),
            "n_related": len(s.get("related", [])),
        }
        if s.get("skill_tree"):
            row["skill_tree"] = s["skill_tree"]
        by_cat[s["category"]].append(row)
    for cat in by_cat:
        by_cat[cat].sort(key=lambda x: -int(x["doc_freq_asr"] or 0))
    return {
        "version": _SKILL_GRAPH.version,
        "total_skills": _SKILL_GRAPH.total,
        "categories": {cat: {"count": len(items), "skills": items[:20]}
                       for cat, items in sorted(by_cat.items())},
    }


def _embedding_job_fallback(query: str) -> dict | None:
    """当 KG L1/L2 都 miss 时，用已常驻的 _PIPELINE 做 embedding 级兜底匹配标准岗位。"""
    if not _PIPELINE_READY or _PIPELINE is None:
        return None
    r = _PIPELINE.match(query, "", include_semantic_expansions=False)
    if r and r.get("standard_id"):
        return {
            "standard_id": r["standard_id"],
            "standard_title": r.get("standard_title", ""),
            "confidence": r.get("confidence", 0),
        }
    return None


@app.post("/api/expand_query")
async def api_expand_query(req: ExpandQueryReq):
    if _SKILL_GRAPH is None:
        raise HTTPException(status_code=503, detail="skill_graph_v1.json 未加载成功")
    fallback = _embedding_job_fallback if req.use_embedding_fallback else None
    res = _do_expand_query(
        raw_query=req.query,
        kg=KG,
        kg_raw=KG_RAW,
        skill_graph=_SKILL_GRAPH,
        include_skill_expansion=req.include_skill_expansion,
        max_skill_hops=req.max_skill_hops,
        embedding_fallback_fn=fallback,
    )
    return res


if __name__ == "__main__":
    import os, uvicorn
    port = int(os.environ.get("PORT", 8901))
    uvicorn.run(app, host="0.0.0.0", port=port)
