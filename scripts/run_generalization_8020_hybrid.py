# -*- coding: utf-8 -*-
"""
离线生成 dataset/generalization_8020_hybrid.json：

与 app.run_generalization 相同的 80/20 变体 holdout（5 轮，seed=42+rd），
在「缩减后的职位 KG」上对比：
  - 纯规则（TitleKnowledgeGraph.match）
  - 混合链路（HybridPipeline：L1/L2 用缩减图，miss 则 Embedding L3）

依赖：本地 Qwen3-Embedding 与 dataset/kg_variants 向量索引（见 embedding_service）。

方法说明：Embedding 索引通常按全量词表构建，与「从图中摘掉 20% 边」并不完全同分布，
hybrid 指标可视为带一定上界；严格无泄漏需对 holdout 变体重建索引。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _collect_variants(kg_raw: dict) -> list[dict]:
    all_variants: list[dict] = []
    for st in kg_raw["standard_titles"]:
        sid, label = st["id"], st["label"]
        for v in st.get("variants", []):
            all_variants.append({"v": v, "sid": sid, "label": label})
        for v in st.get("senior_variants", []):
            all_variants.append({"v": v, "sid": sid, "label": label})
        for _d, vs in st.get("domain_variants", {}).items():
            for v in vs:
                all_variants.append({"v": v, "sid": sid, "label": label})
    return all_variants


def _apply_reduced_kg(pipe, reduced: dict) -> None:
    from app import TitleKnowledgeGraph

    pipe.kg_raw = reduced
    pipe.kg = TitleKnowledgeGraph(reduced)
    pipe.allowed_ids = {st["id"] for st in reduced["standard_titles"]}
    pipe.std_label_by_id = {st["id"]: st["label"] for st in reduced["standard_titles"]}
    pipe.std_category_by_id = {st["id"]: st["category"] for st in reduced["standard_titles"]}
    pipe._kg_context_cache = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5, help="蒙特卡洛轮数（默认 5；调试用 1）")
    args = ap.parse_args()
    n_rounds = max(1, args.rounds)

    from app import TitleKnowledgeGraph, _reduce_kg
    from pipeline import HybridPipeline

    kg_path = ROOT / "knowledge_graph_v2.json"
    kg_raw = json.loads(kg_path.read_text(encoding="utf-8"))
    variants = _collect_variants(kg_raw)
    holdout_ratio = 0.2

    pl = HybridPipeline.from_defaults(llm_backend="embedding")

    rounds_out: list[dict] = []
    for rd in range(n_rounds):
        print(f"[8020-hybrid] round {rd + 1}/{n_rounds} …", flush=True)
        random.seed(42 + rd)
        shuffled = list(variants)
        random.shuffle(shuffled)
        cut = int(len(shuffled) * holdout_ratio)
        holdout, keep = shuffled[:cut], shuffled[cut:]
        keep_set = {item["v"] for item in keep}
        reduced = _reduce_kg(kg_raw, keep_set)

        g = TitleKnowledgeGraph(reduced)
        kg_ok = 0
        for item in holdout:
            if g.match(item["v"]).get("standard_title") == item["label"]:
                kg_ok += 1

        _apply_reduced_kg(pl, reduced)
        hy_ok = 0
        for item in holdout:
            r = pl.match(item["v"], "", include_semantic_expansions=False)
            if r.get("standard_title") == item["label"]:
                hy_ok += 1

        n = len(holdout)
        rounds_out.append({
            "round": rd + 1,
            "holdout": n,
            "kg_correct": kg_ok,
            "kg_accuracy": round(kg_ok / n, 4) if n else 0.0,
            "hybrid_correct": hy_ok,
            "hybrid_accuracy": round(hy_ok / n, 4) if n else 0.0,
        })

    avg_kg = sum(x["kg_accuracy"] for x in rounds_out) / len(rounds_out)
    avg_hy = sum(x["hybrid_accuracy"] for x in rounds_out) / len(rounds_out)

    out = {
        "total_variants": len(variants),
        "holdout_ratio": holdout_ratio,
        "monte_carlo_rounds": n_rounds,
        "rounds": rounds_out,
        "avg_kg_accuracy": round(avg_kg, 4),
        "avg_hybrid_accuracy": round(avg_hy, 4),
        "method_note": (
            "hybrid：缩减图走 L1/L2，未命中则走 Embedding L3；向量索引若按全量词表构建，"
            "相对「摘掉 20% 边」存在乐观偏差，严格评估需对 holdout 重算索引。"
        ),
        "generated_by": "scripts/run_generalization_8020_hybrid.py",
    }
    if n_rounds < 5:
        out["sampling_note"] = (
            f"当前 monte_carlo_rounds={n_rounds}；正式汇报建议默认 5 轮求平均："
            "python scripts/run_generalization_8020_hybrid.py"
        )

    out_path = ROOT / "dataset" / "generalization_8020_hybrid.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
