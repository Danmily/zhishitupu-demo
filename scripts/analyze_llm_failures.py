# -*- coding: utf-8 -*-
"""
从 pipeline_eval_report.csv 聚合 LLM 失败 case，输出：

    dataset/llm_failures_by_std.csv   按期望 standard 分组统计（缺口画像）
    dataset/llm_failures_raw.csv      所有失败样本原文（去重）
    dataset/kg_expand_candidates.csv  建议加进 KG 的候选变体（给下一轮 LLM 优化词表用）

这三份文件可以直接投喂给"让大模型优化词表"的下一轮 prompt，实现:
    1) LLM 看到现有 KG 的盲区在哪
    2) LLM 建议哪些变体该加到哪个标准下
    3) 人工审核后 merge 回 knowledge_graph_v2.json，闭环
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORT_CSV = ROOT / "dataset" / "pipeline_eval_report.csv"
KG_PATH = ROOT / "knowledge_graph_v2.json"
OUT_DIR = ROOT / "dataset"


def _load_report() -> list[dict]:
    if not REPORT_CSV.exists():
        raise FileNotFoundError(f"{REPORT_CSV} 不存在，先跑 scripts/eval_pipeline.py")
    with REPORT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _load_kg_id2label() -> dict[str, str]:
    kg = json.loads(KG_PATH.read_text(encoding="utf-8"))
    return {st["label"]: st["id"] for st in kg["standard_titles"]}


def main() -> None:
    rows = _load_report()
    label2id = _load_kg_id2label()

    bad = [r for r in rows if int(r.get("final_correct", 0)) == 0 and r.get("final_got") is not None]
    llm_bad = [r for r in bad if r.get("final_source") in ("llm", "llm_other", "budget")]

    print(f"[INFO] 报表行数: {len(rows)}")
    print(f"[INFO] 最终错误: {len(bad)}  其中 LLM 经手: {len(llm_bad)}")

    by_std = defaultdict(list)
    for r in llm_bad:
        by_std[r["expected"]].append(r)

    std_summary_rows = []
    for expected_label, items in sorted(by_std.items(), key=lambda x: -len(x[1])):
        llm_guess_counter = Counter(r["final_got"] or "(empty)" for r in items)
        std_summary_rows.append({
            "expected_standard": expected_label,
            "expected_id": label2id.get(expected_label, ""),
            "fail_count": len(items),
            "top_llm_mistakes": "; ".join(
                f"{g} (x{c})" for g, c in llm_guess_counter.most_common(3)
            ),
            "sample_titles": " | ".join(
                sorted({r["input_title"] for r in items})[:6]
            ),
        })

    p1 = OUT_DIR / "llm_failures_by_std.csv"
    with p1.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(std_summary_rows[0].keys()) if std_summary_rows else
                           ["expected_standard", "expected_id", "fail_count", "top_llm_mistakes", "sample_titles"])
        w.writeheader()
        w.writerows(std_summary_rows)
    print(f"[OK] {p1.relative_to(ROOT)}  ({len(std_summary_rows)} rows)")

    seen = set()
    raw_rows = []
    for r in llm_bad:
        key = (r["input_title"], r.get("expected", ""))
        if key in seen:
            continue
        seen.add(key)
        raw_rows.append({
            "input_title": r["input_title"],
            "context": r.get("context", ""),
            "expected": r["expected"],
            "expected_id": label2id.get(r["expected"], ""),
            "llm_got": r.get("final_got", ""),
            "llm_reason": r.get("llm_reason", ""),
            "difficulty": r.get("difficulty", ""),
            "target_level": r.get("target_level", ""),
        })
    p2 = OUT_DIR / "llm_failures_raw.csv"
    with p2.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()) if raw_rows else
                           ["input_title","context","expected","expected_id","llm_got","llm_reason","difficulty","target_level"])
        w.writeheader()
        w.writerows(raw_rows)
    print(f"[OK] {p2.relative_to(ROOT)}  ({len(raw_rows)} rows)")

    candidate_rows = []
    for r in raw_rows:
        candidate_rows.append({
            "candidate_variant": r["input_title"],
            "suggested_standard": r["expected"],
            "suggested_standard_id": r["expected_id"],
            "llm_currently_maps_to": r["llm_got"],
            "evidence_context": r["context"],
            "note": "LLM_failed_case_to_review",
        })
    p3 = OUT_DIR / "kg_expand_candidates.csv"
    with p3.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(candidate_rows[0].keys()) if candidate_rows else
                           ["candidate_variant","suggested_standard","suggested_standard_id",
                            "llm_currently_maps_to","evidence_context","note"])
        w.writeheader()
        w.writerows(candidate_rows)
    print(f"[OK] {p3.relative_to(ROOT)}  ({len(candidate_rows)} rows)")

    print("\n[TOP 期望标准缺口]")
    for r in std_summary_rows[:10]:
        print(f"  {r['expected_standard']:<10} fail={r['fail_count']:>3}  "
              f"top错判: {r['top_llm_mistakes']}")

    print("\n下一步可以把 kg_expand_candidates.csv 喂给 LLM：")
    print("  > 这些是我们现有知识图谱没覆盖到的职位描述。")
    print("  > 请为每一条，给出应加到哪个 standard_id 的 variants/domain_variants 下，")
    print("  > 以及推荐的规范写法。输出 JSON 列表。")


if __name__ == "__main__":
    main()
