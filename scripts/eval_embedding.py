# -*- coding: utf-8 -*-
"""
对比评测：纯 KG (L1/L2/L3) vs KG + Embedding 兜底。

输入：
    dataset/queries.csv

输出：
    dataset/eval_report.csv    - 每条 query 的详细对比
    控制台汇总准确率（总体 + 按 difficulty + 按 target_level）

策略：
    A. 纯 KG：当前 app.py::TitleKnowledgeGraph.match 的 L1/L2/L3 流水线
    B. KG + Embedding：
       - 先跑 L1/L2（高置信度直接返回）
       - 否则使用 Qwen3-Embedding-0.6B 做向量召回（top_k）
       - 若向量 top-1 置信度 >= 阈值，则返回 embedding 结果
       - 否则回退到 L3 图谱推理

重点关注：L3 的 "无法解决 / 错误" case 是否被 Embedding 方案救回。

使用：
    python scripts/eval_embedding.py                # 全量
    python scripts/eval_embedding.py --sample 100   # 快速抽样
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import TitleKnowledgeGraph  # noqa: E402
from embedding_service import EmbeddingKG  # noqa: E402


KG_PATH = ROOT / "knowledge_graph_v2.json"
QUERIES_CSV = ROOT / "dataset" / "queries.csv"
REPORT_CSV = ROOT / "dataset" / "eval_report.csv"

EMB_THRESHOLD = 0.55  # top-1 cosine 需要 >= 该值才信任 embedding 结果


def load_queries(limit: int | None = None) -> list[dict]:
    with QUERIES_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    return rows


def run_pure_kg(kg: TitleKnowledgeGraph, q: dict) -> dict:
    r = kg.match(q["input_title"], q.get("context", ""))
    return {
        "got": r.get("standard_title") or "",
        "level": r.get("level") or "",
        "confidence": r.get("confidence") or 0.0,
    }


def run_kg_plus_embedding(
    kg: TitleKnowledgeGraph, ekg: EmbeddingKG, q: dict
) -> dict:
    title = q["input_title"]
    context = q.get("context", "") or ""

    r1 = kg.level1_exact(title)
    if r1:
        return {
            "got": r1["standard_title"],
            "level": "L1_exact",
            "confidence": r1["confidence"],
            "source": "kg",
        }

    r2 = kg.level2_fuzzy(title)
    if r2 and (r2.get("confidence") or 0) >= 0.75:
        return {
            "got": r2["standard_title"],
            "level": "L2_fuzzy",
            "confidence": r2["confidence"],
            "source": "kg",
        }

    query = title if not context else f"{title} | {context}"
    hits = ekg.search(query, top_k=5)
    if hits and hits[0].score >= EMB_THRESHOLD:
        return {
            "got": hits[0].standard_label,
            "level": "L2_embedding",
            "confidence": round(hits[0].score, 3),
            "source": "embedding",
            "top1_variant": hits[0].variant_name,
        }

    if r2:
        return {
            "got": r2["standard_title"],
            "level": "L2_fuzzy",
            "confidence": r2["confidence"],
            "source": "kg",
        }

    r3 = kg.level3_graph_reasoning(title, context)
    if r3:
        return {
            "got": r3["standard_title"],
            "level": "L3_graph",
            "confidence": r3["confidence"],
            "source": "kg",
        }
    return {"got": "", "level": "L0_miss", "confidence": 0.0, "source": "miss"}


def _agg(stats: dict) -> dict:
    out = {}
    for k, v in stats.items():
        out[k] = {
            **v,
            "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="只跑前 N 条（0=全量）")
    parser.add_argument("--skip-embedding", action="store_true", help="只跑纯 KG，用于冒烟测试")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Embedding 模型目录，默认使用 models/Qwen3-Embedding-0.6B；"
                             "微调后可填 models/Qwen3-Embedding-0.6B-ft")
    parser.add_argument("--index-tag", type=str, default=None,
                        help="索引目录后缀，避免不同模型覆盖（默认按模型目录名自动生成）")
    parser.add_argument("--report-suffix", type=str, default="",
                        help="报表文件名后缀，如 '_ft' 会生成 eval_report_ft.csv")
    args = parser.parse_args()

    kg_raw = json.loads(KG_PATH.read_text(encoding="utf-8"))
    kg = TitleKnowledgeGraph(kg_raw)

    ekg = None
    if not args.skip_embedding:
        print("[INFO] 初始化 Embedding 模型（首次会下载/构建索引，约数分钟）...")
        ekg = EmbeddingKG(model_path=args.model_path, index_tag=args.index_tag)
        ekg.build_index()

    queries = load_queries(limit=args.sample or None)
    print(f"[INFO] 评测 {len(queries)} 条 query（阈值={EMB_THRESHOLD}）")

    rows_out: list[dict] = []
    total_a = {"correct": 0, "total": 0}
    total_b = {"correct": 0, "total": 0}
    by_level_a: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    by_level_b: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    by_diff_a: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    by_diff_b: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    t0 = time.time()
    for i, q in enumerate(queries, 1):
        expected = q["expected_standard_title"]
        diff = q["difficulty"]

        a = run_pure_kg(kg, q)
        a_ok = a["got"] == expected
        total_a["total"] += 1
        total_a["correct"] += int(a_ok)
        by_level_a[a["level"]]["total"] += 1
        by_level_a[a["level"]]["correct"] += int(a_ok)
        by_diff_a[diff]["total"] += 1
        by_diff_a[diff]["correct"] += int(a_ok)

        if ekg is not None:
            b = run_kg_plus_embedding(kg, ekg, q)
            b_ok = b["got"] == expected
            total_b["total"] += 1
            total_b["correct"] += int(b_ok)
            by_level_b[b["level"]]["total"] += 1
            by_level_b[b["level"]]["correct"] += int(b_ok)
            by_diff_b[diff]["total"] += 1
            by_diff_b[diff]["correct"] += int(b_ok)
        else:
            b = {"got": "", "level": "-", "confidence": 0.0, "source": "-"}
            b_ok = False

        rows_out.append({
            "test_id": q["test_id"],
            "input_title": q["input_title"],
            "context": q.get("context", ""),
            "difficulty": diff,
            "target_level": q["target_level"],
            "expected": expected,
            "kg_got": a["got"],
            "kg_level": a["level"],
            "kg_conf": a["confidence"],
            "kg_ok": int(a_ok),
            "emb_got": b["got"],
            "emb_level": b["level"],
            "emb_conf": b["confidence"],
            "emb_source": b.get("source", ""),
            "emb_ok": int(b_ok),
            "rescued": int((not a_ok) and b_ok),
            "regressed": int(a_ok and (not b_ok)),
        })

        if i % 100 == 0:
            print(f"  进度 {i}/{len(queries)}  用时 {time.time()-t0:.1f}s")

    report_path = REPORT_CSV
    if args.report_suffix:
        report_path = REPORT_CSV.with_name(
            REPORT_CSV.stem + args.report_suffix + REPORT_CSV.suffix
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    print("\n" + "=" * 70)
    print("[RESULT] 总体对比")
    print("=" * 70)
    acc_a = total_a["correct"] / total_a["total"] if total_a["total"] else 0
    print(f"  A. 纯 KG            : {total_a['correct']}/{total_a['total']}  acc={acc_a:.4f}")
    if ekg is not None:
        acc_b = total_b["correct"] / total_b["total"] if total_b["total"] else 0
        print(f"  B. KG + Embedding   : {total_b['correct']}/{total_b['total']}  acc={acc_b:.4f}")
        rescued = sum(r["rescued"] for r in rows_out)
        regressed = sum(r["regressed"] for r in rows_out)
        print(f"  Embedding 救回      : +{rescued} 条")
        print(f"  Embedding 回退出错  : -{regressed} 条")
        print(f"  净提升              : {rescued - regressed:+d} 条")

    print("\n[RESULT] 按难度 (A/B 准确率)")
    for d in sorted(set(list(by_diff_a.keys()) + list(by_diff_b.keys()))):
        aa = _agg(by_diff_a).get(d, {})
        bb = _agg(by_diff_b).get(d, {})
        print(
            f"  {d:>6} :  A {aa.get('correct',0)}/{aa.get('total',0)} ({aa.get('accuracy',0):.3f})   "
            f"B {bb.get('correct',0)}/{bb.get('total',0)} ({bb.get('accuracy',0):.3f})"
        )

    print("\n[RESULT] A 按命中 Level")
    for lv, v in _agg(by_level_a).items():
        print(f"  {lv:>14} :  {v['correct']}/{v['total']}  acc={v['accuracy']:.4f}")
    if ekg is not None:
        print("\n[RESULT] B 按命中 Level")
        for lv, v in _agg(by_level_b).items():
            print(f"  {lv:>14} :  {v['correct']}/{v['total']}  acc={v['accuracy']:.4f}")

    print(f"\n[DONE] 详细报表已写入: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
