# -*- coding: utf-8 -*-
"""
从业务技能 Excel 为**多个**语言/框架根技能生成同构五叉子树（爬虫 / 脚本 / 开发 / 测试 / 数据分析）。

对每张根节点：
1. 先用 row_filter 在 skill_name 上筛出「与该栈相关」的行；
2. 再按全局五类分桶 regex 统计词频，取 TOP_N；
3. 开发类用 dev_boost 把该生态词条排在前面（避免全是 Java/Go 岗通用词时淹没本栈信号）。

用法：
  python scripts/ingest_excel_skill_trees.py [path/to.xlsx]
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SKILL_GRAPH = ROOT / "skill_graph_v1.json"
OUT_SNAPSHOT = ROOT / "dataset" / "skill_trees_from_excel.json"

DEFAULT_XLSX = Path(
    r"C:\Users\Damily\Downloads\_SELECT_DISTINCT_tp_key_name_COALESCE_elem_name_elem_AS_skill_na_202604221758.xlsx"
)

TOP_N = 22

# 五类分桶：与业务 Excel 中高频写法对齐（跨语言复用，交集由 row_filter 控制）
BUCKET_DEFS: list[tuple[str, str, str, re.Pattern]] = [
    (
        "cap_crawler",
        "爬虫",
        "采集与反爬、站点/数据抓取、浏览器自动化（非测试向）等",
        re.compile(
            r"(?:网络|网页)?爬虫|反爬|Scrapy|Beautiful\s*Soup|beautifulsoup|"
            r"selenium(?!.*测试)|playwright(?!.*测试)|站点抓取|页面抓取|网页抓取|"
            r"数据抓取(?!.*测试)$"
        ),
    ),
    (
        "cap_script",
        "脚本",
        "Shell/批处理/自动化与运维向脚本、构建脚本等",
        re.compile(
            r"(?:Shell|Bash)脚本?|批处理|自动化脚本|运维脚本|定时脚本|"
            r"python脚本|Python\s*脚本|Gradle\s*脚本?|构建脚本|Makefile|"
            r"Shell\s*编程|Bash(?:\s*开发)?|脚本编写|脚本开发",
            re.I,
        ),
    ),
    (
        "cap_dev",
        "开发",
        "接口/服务/微服务/Web/后端/框架 等工程交付",
        re.compile(
            r"后端开发|Web开发|接口开发|服务端开发|微服务|前后端开发|"
            r"REST(?:ful)?\s*API|gRPC|OpenAPI|GraphQL|BFF|"
            r"Spring|SpringBoot|MyBatis|Django|Flask|FastAPI|Tornado|Gin|"
            r"Node\.?js|Express|Koa|Nest|React|Vue|Angular|Next\.?js",
            re.I,
        ),
    ),
    (
        "cap_test",
        "测试",
        "质量、自动化/接口/性能/回归测试等",
        re.compile(
            r"(?:自动化|接口|单元|集成|性能|压力|回归)?测试|"
            r"\bpytest\b|Pytest|JUnit|Jest|Cypress|测试用例|质量保障|软件测试(?!开发)"
        ),
    ),
    (
        "cap_data",
        "数据分析",
        "数据工程、分析、建模、可视化、数仓/管道 等",
        re.compile(
            r"数据(?:分析|挖掘|科学|工程|处理|开发)|\bETL\b|数据可视化|"
            r"报表|仪表盘|Pandas|NumPy|数据建模|统计分析|"
            r"数仓|数据管道|data\s*pipeline|Hive|Spark|Flink|ClickHouse"
        ),
    ),
]


@dataclass
class RootSpec:
    skill_id: str
    short: str  # 子节点 id 前缀，如 java_cap_dev
    row_filter: re.Pattern
    dev_boost: re.Pattern
    # 可选：各子节点 ties 覆盖（缺省用 DEFAULT_TIES）
    ties: dict[str, list[str]] | None = None


def _default_ties_for(short: str) -> dict[str, list[str]]:
    """子节点边：优先连到图中已有 skill id；可按栈微调。"""
    base = {
        f"{short}_cap_crawler": ["skill_selenium"],
        f"{short}_cap_script": ["skill_linux"],
        f"{short}_cap_dev": ["skill_mysql", "skill_redis"],
        f"{short}_cap_test": ["skill_自动化测试", "skill_接口测试", "skill_功能测试"],
        f"{short}_cap_data": ["skill_pandas", "skill_数据分析", "skill_pytorch"],
    }
    if short == "py":
        base[f"{short}_cap_dev"] = ["skill_vue", "skill_react", "skill_mysql", "skill_redis"]
    elif short == "java":
        base[f"{short}_cap_dev"] = ["skill_spring", "skill_mysql", "skill_redis", "skill_kafka"]
    elif short in ("js", "ts"):
        base[f"{short}_cap_dev"] = ["skill_vue", "skill_react", "skill_node_js", "skill_mysql"]
    elif short == "go":
        base[f"{short}_cap_dev"] = ["skill_微服务", "skill_redis", "skill_mysql", "skill_kafka"]
    elif short == "rust":
        base[f"{short}_cap_dev"] = ["skill_微服务", "skill_redis", "skill_kafka"]
    elif short == "cpp":
        base[f"{short}_cap_dev"] = ["skill_性能优化", "skill_linux", "skill_docker"]
    return base


# 在 skill_graph 中必须存在的 id；不存在的会跳过
ROOTS: list[RootSpec] = [
    RootSpec(
        "skill_python",
        "py",
        re.compile(
            r"Python|Django|Flask|FastAPI|PyPy|Pandas|NumPy|Scrapy|BeautifulSoup|"
            r"PyTorch|\.py\b|Gunicorn|Uvicorn|WSGI|ASGI|Celery|Airflow",
            re.I,
        ),
        re.compile(
            r"Python|Django|Flask|FastAPI|\bTornado\b|Gunicorn|Uvicorn|WSGI|ASGI|"
            r"PyTorch|Celery|BFF|GraphQL|OpenAPI|Swagger|\bMCP\b",
            re.I,
        ),
    ),
    RootSpec(
        "skill_java",
        "java",
        re.compile(
            r"(?<!Script)(?<!\.)\bJava(?!Script)\b|JVM|J2EE|Spring|MyBatis|Hibernate|"
            r"Gradle|Maven|Tomcat|Jetty|SpringBoot|SpringCloud|Kotlin",
            re.I,
        ),
        re.compile(
            r"Java|Spring|MyBatis|Hibernate|Gradle|Maven|JVM|"
            r"SpringBoot|SpringCloud|Dubbo|Quarkus|Jakarta|Kafka(?!.Go)",
            re.I,
        ),
    ),
    RootSpec(
        "skill_go",
        "go",
        re.compile(r"Golang|Go语言|\bGo\s*(?:微服务|开发|语言)?|\bGin\b|Beego|Fiber|GORM|"
                   r"etcd|Consul(?!.Java)", re.I),
        re.compile(r"Gin|Golang|Goroutine|gRPC|Beego|Fiber|GORM|Kratos|micro", re.I),
    ),
    RootSpec(
        "skill_javascript",
        "js",
        re.compile(
            r"JavaScript|\bES6\b|Node\.?js|React|Vue\.|Angular|webpack|Vite|"
            r"npm|yarn|pnpm|Express|Koa|Next\.js|Nuxt|jQuery(?!.Java)",
            re.I,
        ),
        re.compile(
            r"JavaScript|React|Vue|Angular|Node|Express|Koa|Next\.?js|Nuxt|"
            r"webpack|Vite|npm|Zustand|Redux|MobX|Three\.js",
            re.I,
        ),
    ),
    RootSpec(
        "skill_typescript",
        "ts",
        re.compile(
            r"TypeScript|\.tsx?\b|tRPC|Prisma|NestJS|Deno(?!.Java)", re.I
        ),
        re.compile(
            r"TypeScript|Nest|tRPC|Prisma|Deno|Zod|Next\.?js|tsc\b", re.I
        ),
    ),
    RootSpec(
        "skill_rust",
        "rust",
        re.compile(
            r"Rust|Cargo|tokio|actix|axum|Rustls|WASM|wasm|Rocket\.rs", re.I
        ),
        re.compile(
            r"Rust|Cargo|tokio|actix|axum|serde|WASM|wasm", re.I
        ),
    ),
    RootSpec(
        "skill_c",
        "cpp",
        re.compile(
            r"C\+\+|CUDA|OpenCV|Qt|Unreal|LLVM|CMake|Catch2|GoogleTest|"
            r"多线程|智能指针|性能优化(?!.Python)",
            re.I,
        ),
        re.compile(
            r"C\+\+|CUDA|OpenCV|Qt|Unreal|LLVM|CMake|CMakeLists|GTest|"
            r"多线程|智能指针|内存管理",
            re.I,
        ),
    ),
]


def _norm(s: str) -> str:
    t = (s or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _ordered_bucket(ser: pd.Series, spec: RootSpec, pat: re.Pattern, bid_suffix: str) -> list[str]:
    """先取「行属于该语言 ∧ 分桶」的高频；过少时用全表同分桶补全，仍保持语言行优先序。"""
    m_b = ser.map(lambda t: bool(pat.search(t)))
    m_r = ser.map(lambda t: bool(spec.row_filter.search(t)))
    pref = ser[m_b & m_r]
    loose = ser[m_b]
    if len(pref) >= 5:
        base_order = list(dict.fromkeys(pref.value_counts().index.tolist()))
    else:
        base_order = list(
            dict.fromkeys(
                pref.value_counts().index.tolist() + loose.value_counts().index.tolist()
            )
        )
    if bid_suffix != "cap_dev":
        return [str(k) for k in base_order[:TOP_N] if k]
    boosted = [k for k in base_order if k and spec.dev_boost.search(str(k))]
    rest = [k for k in base_order if k and k not in boosted]
    merged = boosted + rest
    return [str(k) for k in merged[:TOP_N] if k]


def mine_for_root(ser: pd.Series, spec: RootSpec) -> dict[str, list[str]]:
    """ser: 全表 skill_name 列（已 norm）。"""
    out: dict[str, list[str]] = {}
    for bid_suffix, _label, _desc, pat in BUCKET_DEFS:
        key = f"{spec.short}_{bid_suffix}"
        out[key] = _ordered_bucket(ser, spec, pat, bid_suffix)
    return out


def _cross_edges(prefix: str) -> list[dict]:
    a, b, c, d, e = (
        f"{prefix}_cap_crawler",
        f"{prefix}_cap_script",
        f"{prefix}_cap_dev",
        f"{prefix}_cap_test",
        f"{prefix}_cap_data",
    )
    return [
        {"edge": "undirected", "between": [a, b], "note": "采集与脚本编排、清洗常联动。"},
        {"edge": "undirected", "between": [c, d], "note": "研发交付中开发与测试强耦合。"},
        {"edge": "undirected", "between": [e, c], "note": "数据服务与业务接口常联调。"},
    ]


def patch_all(mined: dict[str, dict[str, list[str]]], graph_path: Path) -> int:
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    skills = data.get("skills", [])
    by_id = {s["id"]: s for s in skills}
    n_ok = 0
    for spec in ROOTS:
        node = by_id.get(spec.skill_id)
        if not node:
            print(f"[WARN] 图中无 {spec.skill_id}，跳过", file=sys.stderr)
            continue
        label = node.get("label", spec.skill_id)
        bag = mined.get(spec.skill_id) or {}
        children: list[dict] = []
        ties_map = {**_default_ties_for(spec.short), **(spec.ties or {})}
        for bid_suffix, b_label, b_desc, _ in BUCKET_DEFS:
            cid = f"{spec.short}_{bid_suffix}"
            terms = bag.get(cid) or []
            aliases = [b_label] + [t for t in terms if t and t not in b_label][ :17]
            ch: dict = {
                "id": cid,
                "label": b_label,
                "edge": "directed",
                "description": b_desc,
                "aliases": aliases,
                "excel_top_terms": terms,
            }
            tids = ties_map.get(cid)
            if tids:
                ch["ties"] = tids
            children.append(ch)
        node["skill_tree"] = {
            "description": f"{label} 为根节点的五类工程方向子树（业务词表按行筛选 + 分桶频次摘取）。",
            "source_excel": "key_name, skill_name (user export)",
            "children": children,
            "cross_edges": _cross_edges(spec.short),
        }
        n_ok += 1
    if "ingest_meta" not in data:
        data["ingest_meta"] = {}
    data["ingest_meta"]["skill_trees_excel"] = str(OUT_SNAPSHOT)
    data["ingest_meta"].pop("python_tree_excel", None)
    graph_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return n_ok


def run_ingest(xlsx: Path) -> tuple[int, dict[str, dict[str, list[str]]]]:
    df = pd.read_excel(xlsx)
    if "skill_name" not in df.columns:
        raise SystemExit("Excel 需含列 skill_name")
    ser = df["skill_name"].map(lambda x: _norm(str(x) if x == x else ""))
    ser = ser[ser.str.len() > 0]

    mined: dict[str, dict[str, list[str]]] = {}
    for spec in ROOTS:
        mined[spec.skill_id] = mine_for_root(ser, spec)

    OUT_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    OUT_SNAPSHOT.write_text(
        json.dumps(mined, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    n = patch_all(mined, SKILL_GRAPH)
    return n, mined


def main() -> int:
    xlsx = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_XLSX
    if not xlsx.exists():
        print(f"未找到 Excel：{xlsx}", file=sys.stderr)
        return 1
    n, _ = run_ingest(xlsx)
    print(f"[OK] 已更新 {n} 个根技能的 skill_tree → {SKILL_GRAPH}")
    print(f"     快照: {OUT_SNAPSHOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
