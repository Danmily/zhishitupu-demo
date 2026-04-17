# -*- coding: utf-8 -*-
"""
混合 Pipeline 评测：KG 规则 (L1/L2) + LLM 泛化 (L3)。

重点关注领导在意的两件事：
    1. L3 LLM 在"KG 解决不了的反例"上的救回率
    2. LLM 不给出正确答案时的失败原因分布（为后续优化词表提供线索）

流程：
    1. 先跑纯 KG 基线（只 L1/L2，不掺 L3 LLM），收集 "other / miss" 的 case
    2. 仅在失败的 case 上调用 LLM，最大化 LLM 效率 & 成本
    3. 计算混合 pipeline 最终指标
    4. 导出反例分析 CSV

用法：
    # Embedding 后端（CPU 可跑，快）
    python scripts/eval_pipeline.py --backend embedding --sample 300

    # 本地 Qwen3-0.6B 生成模型（需先 download_qwen_llm.py；CPU 慢）
    python scripts/eval_pipeline.py --backend local --sample 50

    # OpenAI 兼容 API（推荐，DashScope qwen3-0.6b）
    $env:LLM_API_KEY='sk-xxx'
    $env:LLM_MODEL='qwen3-0.6b'
    python scripts/eval_pipeline.py --backend api --sample 300

    # 只跑 KG 基线（跳过 LLM），看反例池
    python scripts/eval_pipeline.py --no-llm
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
from llm_engine import build_engine  # noqa: E402
from pipeline import HybridPipeline  # noqa: E402

KG_PATH = ROOT / "knowledge_graph_v2.json"
QUERIES_CSV = ROOT / "dataset" / "queries.csv"


def load_queries(limit: int | None = None) -> list[dict]:
    with QUERIES_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    return rows


def _kg_only_match(kg: TitleKnowledgeGraph, title: str, l2_thr: float) -> dict:
    r1 = kg.level1_exact(title)
    if r1:
        return {"got": r1["standard_title"], "level": "L1_exact",
                "confidence": r1.get("confidence", 1.0), "source": "kg"}
    r2 = kg.level2_fuzzy(title)
    if r2 and (r2.get("confidence") or 0) >= l2_thr:
        return {"got": r2["standard_title"], "level": "L2_fuzzy",
                "confidence": r2.get("confidence", 0), "source": "kg"}
    return {"got": "", "level": "L0_kg_miss",
            "confidence": (r2.get("confidence") if r2 else 0), "source": "none"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="embedding",
                        choices=["embedding", "local", "api", "none"])
    parser.add_argument("--sample", type=int, default=0, help="前 N 条（0=全量）")
    parser.add_argument("--l2-threshold", type=float, default=0.75)
    parser.add_argument("--llm-max-calls", type=int, default=0,
                        help="LLM 最多调用次数，0=不限。用于 API 成本控制")
    parser.add_argument("--no-llm", action="store_true",
                        help="只跑 KG 基线，不调 LLM（用于先看反例池）")
    parser.add_argument("--model-path", type=str, default=None,
                        help="local 后端：Qwen3-0.6B 目录；api 后端：无需填")
    parser.add_argument("--report", type=str, default="dataset/pipeline_eval_report.csv")
    parser.add_argument("--miss-csv", type=str, default="dataset/pipeline_miss_cases.csv",
                        help="KG L1/L2 未解决的反例清单（即 LLM 需要泛化的那批）")
    args = parser.parse_args()

    kg_raw = json.loads(KG_PATH.read_text(encoding="utf-8"))
    kg = TitleKnowledgeGraph(kg_raw)
    allowed_ids = {st["id"] for st in kg_raw["standard_titles"]}
    std_label_by_id = {st["id"]: st["label"] for st in kg_raw["standard_titles"]}

    llm = None
    if not args.no_llm and args.backend != "none":
        kwargs = {}
        if args.backend == "local" and args.model_path:
            kwargs["model_path"] = args.model_path
        print(f"[INFO] 初始化 LLM 后端: {args.backend}")
        llm = build_engine(args.backend, **kwargs)

    pipeline = HybridPipeline(kg, kg_raw, llm=llm, l2_confidence_threshold=args.l2_threshold)

    queries = load_queries(limit=args.sample or None)
    print(f"[INFO] 评测 {len(queries)} 条 query；l2_threshold={args.l2_threshold}")

    report_rows: list[dict] = []
    miss_rows: list[dict] = []

    total = 0
    kg_solved = 0
    kg_correct = 0
    kg_missed = 0
    llm_called = 0
    llm_correct = 0
    llm_wrong = 0
    llm_other = 0
    llm_budget_skipped = 0
    final_correct = 0

    by_target_level: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0,
                                                            "kg": 0, "llm": 0})
    llm_err_examples: list[dict] = []

    t0 = time.time()
    for i, q in enumerate(queries, 1):
        total += 1
        expected = q["expected_standard_title"]
        target = q["target_level"]
        by_target_level[target]["total"] += 1

        kg_r = _kg_only_match(kg, q["input_title"], args.l2_threshold)
        kg_ok = kg_r["got"] == expected

        if kg_r["source"] == "kg":
            kg_solved += 1
            if kg_ok:
                kg_correct += 1
                final_correct += 1
                by_target_level[target]["correct"] += 1
                by_target_level[target]["kg"] += 1
        else:
            kg_missed += 1
            miss_rows.append({
                "test_id": q["test_id"],
                "input_title": q["input_title"],
                "context": q.get("context", ""),
                "difficulty": q["difficulty"],
                "target_level": target,
                "expected": expected,
                "kg_level": kg_r["level"],
                "kg_confidence": kg_r["confidence"],
            })

        if args.no_llm or llm is None:
            final_got = kg_r["got"] if kg_r["source"] == "kg" else ""
            final_source = kg_r["source"]
            final_level = kg_r["level"]
            reason = ""
        elif kg_r["source"] == "kg":
            final_got = kg_r["got"]
            final_source = "kg"
            final_level = kg_r["level"]
            reason = ""
        else:
            if args.llm_max_calls and llm_called >= args.llm_max_calls:
                llm_budget_skipped += 1
                final_got, final_source, final_level = "", "budget", "L0_budget"
                reason = f"llm_budget={args.llm_max_calls}_reached"
            else:
                llm_called += 1
                llm_res = llm.predict(
                    title=q["input_title"],
                    context=q.get("context", ""),
                    kg_context_str=pipeline.kg_context,
                    allowed_ids=allowed_ids,
                    std_label_by_id=std_label_by_id,
                )
                sid = llm_res.get("standard_id")
                reason = llm_res.get("reason", "")
                if sid and sid in allowed_ids:
                    final_got = llm_res.get("standard_title") or std_label_by_id.get(sid, "")
                    final_source = "llm"
                    final_level = f"L3_llm_{llm.name}"
                    if final_got == expected:
                        llm_correct += 1
                        final_correct += 1
                        by_target_level[target]["correct"] += 1
                        by_target_level[target]["llm"] += 1
                    else:
                        llm_wrong += 1
                        if len(llm_err_examples) < 20:
                            llm_err_examples.append({
                                "title": q["input_title"],
                                "expected": expected,
                                "llm_got": final_got,
                                "reason": reason,
                            })
                else:
                    llm_other += 1
                    final_got, final_source, final_level = "", "llm_other", "L0_llm_other"

        report_rows.append({
            "test_id": q["test_id"],
            "input_title": q["input_title"],
            "context": q.get("context", ""),
            "difficulty": q["difficulty"],
            "target_level": target,
            "expected": expected,
            "kg_got": kg_r["got"],
            "kg_level": kg_r["level"],
            "kg_conf": kg_r["confidence"],
            "final_got": final_got,
            "final_source": final_source,
            "final_level": final_level,
            "final_correct": int(final_got == expected and bool(final_got)),
            "llm_reason": reason,
        })

        if i % 100 == 0:
            print(f"  进度 {i}/{len(queries)}  用时 {time.time()-t0:.1f}s  "
                  f"llm_called={llm_called}  final_acc={final_correct/i:.3f}")

    report_path = ROOT / args.report
    miss_path = ROOT / args.miss_csv
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        w.writeheader()
        w.writerows(report_rows)

    if miss_rows:
        with miss_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(miss_rows[0].keys()))
            w.writeheader()
            w.writerows(miss_rows)

    print("\n" + "=" * 72)
    print("[PIPELINE RESULT]")
    print("=" * 72)
    print(f"  total queries            : {total}")
    print(f"  KG L1/L2 命中             : {kg_solved}  ({kg_solved/total:.1%})")
    print(f"     └ 其中正确            : {kg_correct}  ({kg_correct/max(kg_solved,1):.1%})")
    print(f"  KG 未解决（送 LLM）       : {kg_missed}")
    print(f"  LLM 实际调用              : {llm_called}")
    if llm_called:
        print(f"     ├ 救回正确            : {llm_correct}  ({llm_correct/llm_called:.1%})")
        print(f"     ├ 挑错了标准          : {llm_wrong}")
        print(f"     └ 输出 other/非法     : {llm_other}")
    if llm_budget_skipped:
        print(f"  LLM 预算用尽跳过          : {llm_budget_skipped}")
    final_acc = final_correct / total
    print(f"  最终整条 pipeline 准确率   : {final_correct}/{total} = {final_acc:.4f}")

    pure_kg_acc = kg_correct / total
    print(f"  纯 KG 基线准确率           : {pure_kg_acc:.4f}")
    print(f"  LLM 带来的绝对提升         : {final_acc - pure_kg_acc:+.4f}  "
          f"({(final_acc-pure_kg_acc)*100:+.2f} 个百分点)")

    print("\n按目标难度 (target_level):")
    for lv in sorted(by_target_level):
        v = by_target_level[lv]
        acc = v["correct"] / max(v["total"], 1)
        print(f"  {lv:>4}:  {v['correct']}/{v['total']} ({acc:.3f})   "
              f"kg={v['kg']}  llm={v['llm']}")

    if llm_err_examples:
        print("\n[LLM 错误典型案例 (最多 20 条)]")
        for e in llm_err_examples:
            print(f"  - '{e['title']}'  期望={e['expected']}  LLM={e['llm_got']}  ({e['reason']})")

    print(f"\n[DONE] 报表:")
    print(f"  {report_path.relative_to(ROOT)}")
    if miss_rows:
        print(f"  {miss_path.relative_to(ROOT)}  ({len(miss_rows)} 条 KG 未解决反例)")


if __name__ == "__main__":
    main()
