# -*- coding: utf-8 -*-
"""
将现有的 JSON 测试集 / 知识图谱转换为 agent-friendly 的 CSV 格式。

产出文件（dataset/ 目录下）：
    1. queries.csv         - 测试问题主表（L1/L2/L3 均覆盖）
    2. kg_standards.csv    - 标准职称（带分类与关联技能）
    3. kg_variants.csv     - 变体 -> 标准 的映射边（包含 variant / senior_variant / domain_variant）
    4. kg_cross_refs.csv   - 标准职称之间的交叉引用（晋升/协作/同族）
    5. kg_dimensions.csv   - 维度枚举（seniority / domain / scope / tech_stack）

使用：
    python scripts/convert_to_csv.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
KG_PATH = ROOT / "knowledge_graph_v2.json"
TEST_PATH = ROOT / "test_cases_v2.json"
OUT_DIR = ROOT / "dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(header)
        w.writerows(rows)
    print(f"[OK] {path.relative_to(ROOT)}  ({len(rows)} rows)")


def convert_queries(test_cases: list[dict]) -> None:
    header = [
        "test_id",
        "category",
        "difficulty",
        "target_level",
        "input_title",
        "context",
        "expected_standard_title",
        "expected_standard_id",
        "expected_seniority",
        "expected_domain",
    ]
    rows: list[list] = []
    for tc in test_cases:
        exp = tc.get("expected_output") or {}
        rows.append([
            tc.get("test_id", ""),
            tc.get("category", ""),
            tc.get("difficulty", ""),
            tc.get("target_level", ""),
            tc.get("input_title", ""),
            tc.get("context", "") or "",
            exp.get("standard_title") or "",
            exp.get("standard_id") or "",
            exp.get("seniority") or "",
            exp.get("domain") or "",
        ])
    _write_csv(OUT_DIR / "queries.csv", header, rows)


def convert_kg_standards(kg: dict) -> None:
    header = ["standard_id", "label", "category", "related_skills"]
    rows: list[list] = []
    for st in kg.get("standard_titles", []):
        skills = "|".join(st.get("related_skills", []) or [])
        rows.append([st["id"], st["label"], st.get("category", ""), skills])
    _write_csv(OUT_DIR / "kg_standards.csv", header, rows)


def convert_kg_variants(kg: dict) -> None:
    header = [
        "variant_id",
        "variant_name",
        "standard_id",
        "standard_label",
        "variant_type",
        "domain",
        "seniority_hint",
    ]
    rows: list[list] = []
    vid = 0

    def _emit(variant: str, st: dict, vtype: str, domain: str = "", seniority: str = "") -> None:
        nonlocal vid
        vid += 1
        rows.append([
            f"V{vid:05d}",
            variant,
            st["id"],
            st["label"],
            vtype,
            domain,
            seniority,
        ])

    for st in kg.get("standard_titles", []):
        _emit(st["label"], st, "canonical")

        for v in st.get("variants", []) or []:
            _emit(v, st, "variant")

        for v in st.get("senior_variants", []) or []:
            _emit(v, st, "senior_variant")

        for domain, vs in (st.get("domain_variants", {}) or {}).items():
            for v in vs:
                _emit(v, st, "domain_variant", domain=domain)

    _write_csv(OUT_DIR / "kg_variants.csv", header, rows)


def convert_kg_cross_refs(kg: dict) -> None:
    header = ["from_id", "to_id", "type", "description"]
    rows = [
        [cr.get("from", ""), cr.get("to", ""), cr.get("type", ""), cr.get("desc", "")]
        for cr in kg.get("cross_references", [])
    ]
    _write_csv(OUT_DIR / "kg_cross_refs.csv", header, rows)


def convert_kg_dimensions(kg: dict) -> None:
    header = ["dimension_key", "dimension_label", "value"]
    rows: list[list] = []
    for dim_key, dim in (kg.get("dimensions", {}) or {}).items():
        label = dim.get("label", "")
        for v in dim.get("values", []) or []:
            rows.append([dim_key, label, v])
    _write_csv(OUT_DIR / "kg_dimensions.csv", header, rows)


def main() -> None:
    kg = json.loads(KG_PATH.read_text(encoding="utf-8"))
    tests = json.loads(TEST_PATH.read_text(encoding="utf-8"))

    print(f"[INFO] 加载 KG: {len(kg.get('standard_titles', []))} 个标准职称")
    print(f"[INFO] 加载测试集: {len(tests)} 条记录")
    print(f"[INFO] 输出目录: {OUT_DIR}")
    print("-" * 60)

    convert_queries(tests)
    convert_kg_standards(kg)
    convert_kg_variants(kg)
    convert_kg_cross_refs(kg)
    convert_kg_dimensions(kg)

    print("-" * 60)
    print("[DONE] 全部 CSV 文件已生成于 dataset/ 目录")


if __name__ == "__main__":
    main()
