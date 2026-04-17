# -*- coding: utf-8 -*-
"""
一键对比：未微调 base vs LoRA 微调后 ft。

流程：
    1. 分别用 base / ft 模型为 KG 变体构建向量索引（独立缓存，互不覆盖）
    2. 在 queries.csv 上跑完整 KG+Embedding pipeline
    3. 汇总输出：总体准确率、L3 救回数、回退错误数、按难度分布
    4. 生成 dataset/eval_report_base.csv 与 dataset/eval_report_ft.csv
    5. 生成 dataset/compare_ft_summary.csv（关键指标对比）

使用：
    python scripts/compare_ft.py                 # 全量 1500 query
    python scripts/compare_ft.py --sample 500
    python scripts/compare_ft.py --base-only     # 只跑 base
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
from scripts.eval_embedding import (  # noqa: E402
    load_queries, run_pure_kg, run_kg_plus_embedding, EMB_THRESHOLD,
)

KG_PATH = ROOT / "knowledge_graph_v2.json"
BASE_MODEL = ROOT / "models" / "Qwen3-Embedding-0.6B"
FT_MODEL = ROOT / "models" / "Qwen3-Embedding-0.6B-ft"
REPORT_DIR = ROOT / "dataset"


def run_pipeline(tag: str, model_path: Path, queries: list[dict], kg: TitleKnowledgeGraph) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_path}")

    print(f"\n{'='*70}\n[RUN {tag}] {model_path}\n{'='*70}")
    t0 = time.time()
    ekg = EmbeddingKG(model_path=str(model_path), index_tag=tag)
    ekg.build_index()
    print(f"[RUN {tag}] 索引就绪，耗时 {time.time()-t0:.1f}s")

    rows_out: list[dict] = []
    total_a = {"correct": 0, "total": 0}
    total_b = {"correct": 0, "total": 0}
    by_diff_b: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    by_target_b: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    t1 = time.time()
    for i, q in enumerate(queries, 1):
        expected = q["expected_standard_title"]
        diff = q["difficulty"]
        target = q["target_level"]

        a = run_pure_kg(kg, q)
        b = run_kg_plus_embedding(kg, ekg, q)
        a_ok = a["got"] == expected
        b_ok = b["got"] == expected

        total_a["total"] += 1
        total_a["correct"] += int(a_ok)
        total_b["total"] += 1
        total_b["correct"] += int(b_ok)
        by_diff_b[diff]["total"] += 1
        by_diff_b[diff]["correct"] += int(b_ok)
        by_target_b[target]["total"] += 1
        by_target_b[target]["correct"] += int(b_ok)

        rows_out.append({
            "test_id": q["test_id"],
            "input_title": q["input_title"],
            "context": q.get("context", ""),
            "difficulty": diff,
            "target_level": target,
            "expected": expected,
            "kg_got": a["got"],
            "kg_ok": int(a_ok),
            "emb_got": b["got"],
            "emb_level": b["level"],
            "emb_conf": b["confidence"],
            "emb_source": b.get("source", ""),
            "emb_ok": int(b_ok),
            "rescued": int((not a_ok) and b_ok),
            "regressed": int(a_ok and (not b_ok)),
        })
        if i % 200 == 0:
            print(f"  [{tag}] 进度 {i}/{len(queries)}  {time.time()-t1:.1f}s")

    report_path = REPORT_DIR / f"eval_report_{tag}.csv"
    with report_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"[RUN {tag}] 报表: {report_path.relative_to(ROOT)}  总耗时 {time.time()-t0:.1f}s")

    acc_a = total_a["correct"] / total_a["total"]
    acc_b = total_b["correct"] / total_b["total"]
    rescued = sum(r["rescued"] for r in rows_out)
    regressed = sum(r["regressed"] for r in rows_out)

    return {
        "tag": tag,
        "model_path": str(model_path),
        "total": total_b["total"],
        "kg_only_acc": round(acc_a, 4),
        "kg_plus_emb_acc": round(acc_b, 4),
        "rescued": rescued,
        "regressed": regressed,
        "net_gain": rescued - regressed,
        "by_difficulty": {
            d: round(v["correct"] / v["total"], 4) if v["total"] else 0
            for d, v in by_diff_b.items()
        },
        "by_target_level": {
            t: round(v["correct"] / v["total"], 4) if v["total"] else 0
            for t, v in by_target_b.items()
        },
        "rows": rows_out,
    }


def print_compare(base: dict, ft: dict | None) -> None:
    print("\n" + "=" * 72)
    print("[COMPARE]  未微调 base  vs  LoRA 微调后 ft")
    print("=" * 72)

    def _row(k: str, a, b):
        print(f"  {k:<26}{str(a):>16}{str(b):>16}")

    _row("", "base", "ft" if ft else "(未提供)")
    _row("total queries", base["total"], ft["total"] if ft else "-")
    _row("pure KG acc", f"{base['kg_only_acc']:.4f}",
         f"{ft['kg_only_acc']:.4f}" if ft else "-")
    _row("KG+Embedding acc", f"{base['kg_plus_emb_acc']:.4f}",
         f"{ft['kg_plus_emb_acc']:.4f}" if ft else "-")
    _row("Embedding 救回", f"+{base['rescued']}",
         f"+{ft['rescued']}" if ft else "-")
    _row("Embedding 回退错", f"-{base['regressed']}",
         f"-{ft['regressed']}" if ft else "-")
    _row("net gain", f"{base['net_gain']:+d}",
         f"{ft['net_gain']:+d}" if ft else "-")

    if ft:
        delta = ft["kg_plus_emb_acc"] - base["kg_plus_emb_acc"]
        print(f"\n  -> 微调带来 KG+Embedding 准确率提升: {delta:+.4f} "
              f"({delta*100:+.2f} 个百分点)")

    print("\n按难度:")
    diffs = sorted(set(list(base["by_difficulty"].keys()) +
                       (list(ft["by_difficulty"].keys()) if ft else [])))
    for d in diffs:
        ba = base["by_difficulty"].get(d, 0)
        fa = ft["by_difficulty"].get(d, 0) if ft else None
        if ft:
            print(f"  {d:>8}:  base={ba:.4f}   ft={fa:.4f}   Δ={fa-ba:+.4f}")
        else:
            print(f"  {d:>8}:  base={ba:.4f}")

    print("\n按目标 Level:")
    lvls = sorted(set(list(base["by_target_level"].keys()) +
                      (list(ft["by_target_level"].keys()) if ft else [])))
    for lv in lvls:
        ba = base["by_target_level"].get(lv, 0)
        fa = ft["by_target_level"].get(lv, 0) if ft else None
        if ft:
            print(f"  {lv:>6}:  base={ba:.4f}   ft={fa:.4f}   Δ={fa-ba:+.4f}")
        else:
            print(f"  {lv:>6}:  base={ba:.4f}")


def write_summary(base: dict, ft: dict | None, out_path: Path) -> None:
    fields = [
        "metric", "base", "ft", "delta",
    ]
    rows = []

    def _add(metric: str, b, f):
        if f is None:
            rows.append({"metric": metric, "base": b, "ft": "", "delta": ""})
        else:
            try:
                d = round(float(f) - float(b), 4)
            except Exception:
                d = ""
            rows.append({"metric": metric, "base": b, "ft": f, "delta": d})

    _add("total_queries", base["total"], ft["total"] if ft else None)
    _add("pure_kg_acc", base["kg_only_acc"], ft["kg_only_acc"] if ft else None)
    _add("kg_plus_emb_acc", base["kg_plus_emb_acc"], ft["kg_plus_emb_acc"] if ft else None)
    _add("embedding_rescued", base["rescued"], ft["rescued"] if ft else None)
    _add("embedding_regressed", base["regressed"], ft["regressed"] if ft else None)
    _add("net_gain", base["net_gain"], ft["net_gain"] if ft else None)

    if ft:
        for d in sorted(set(list(base["by_difficulty"].keys()) + list(ft["by_difficulty"].keys()))):
            _add(f"acc_diff_{d}", base["by_difficulty"].get(d, 0), ft["by_difficulty"].get(d, 0))
        for lv in sorted(set(list(base["by_target_level"].keys()) + list(ft["by_target_level"].keys()))):
            _add(f"acc_target_{lv}", base["by_target_level"].get(lv, 0), ft["by_target_level"].get(lv, 0))

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[DONE] summary CSV: {out_path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="评测前 N 条（0=全量）")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--ft-only", action="store_true")
    parser.add_argument("--base-path", type=str, default=str(BASE_MODEL))
    parser.add_argument("--ft-path", type=str, default=str(FT_MODEL))
    args = parser.parse_args()

    print(f"[INFO] EMB_THRESHOLD={EMB_THRESHOLD}")
    kg_raw = json.loads(KG_PATH.read_text(encoding="utf-8"))
    kg = TitleKnowledgeGraph(kg_raw)
    queries = load_queries(limit=args.sample or None)
    print(f"[INFO] 评测 queries: {len(queries)}")

    base = None
    ft = None

    if not args.ft_only:
        base = run_pipeline("base", Path(args.base_path), queries, kg)

    if not args.base_only:
        if not Path(args.ft_path).exists():
            print(f"\n[WARN] 微调模型不存在: {args.ft_path}")
            print("       请先运行: python scripts/finetune_qwen_embedding.py")
        else:
            ft = run_pipeline("ft", Path(args.ft_path), queries, kg)

    if base is not None:
        print_compare(base, ft)
        write_summary(base, ft, REPORT_DIR / "compare_ft_summary.csv")


if __name__ == "__main__":
    main()
