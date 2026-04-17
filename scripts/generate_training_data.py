# -*- coding: utf-8 -*-
"""
为 Qwen3-Embedding-0.6B 微调生成训练数据（约 10000 条三元组）。

输出：
    dataset/embedding_train.csv       ~90% 训练
    dataset/embedding_eval.csv        ~10% 验证
    dataset/embedding_train.jsonl     与 CSV 等价的 JSONL（便于 datasets/transformers 加载）

每条记录字段：
    pair_id           样本 ID
    anchor            "脏" 输入（query 端）
    positive          正确对应的标准职称 label
    positive_id       正样本 standard_id
    hard_negative     同类别但不同标准的 variant（困难负样本）
    hard_negative_id  困难负样本 standard_id
    random_negative   跨类别随机负样本
    random_negative_id
    category          正样本所属类别
    generation_type   生成模板类别（用于数据分析 / 消融）

生成策略（7 类模板）：
    1. T1_raw          原始 variant（含 canonical / senior / domain variant）
    2. T2_seniority    "高级/资深/首席/..." + base（从 seniority 维度）
    3. T3_domain       "AI/B端/C端/金融/..." + base（从 domain 维度）
    4. T4_colloquial   "做/搞/负责 X 的（人/哥们/同事）" 等口语化
    5. T5_en_zh_mix    中英混写、大小写扰动、空格扰动
    6. T6_typo         常见简拼/错字（轻微扰动）
    7. T7_context      variant + 上下文（"在字节做 X"/"带 10 人团队"/"对标 xxx"）

使用：
    python scripts/generate_training_data.py
    python scripts/generate_training_data.py --target 15000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KG_PATH = ROOT / "knowledge_graph_v2.json"
OUT_DIR = ROOT / "dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
#  模板库
# ═══════════════════════════════════════════════════════════

SENIORITY_PREFIXES = [
    "高级", "资深", "首席", "副", "初级", "中级",
    "Senior ", "Junior ", "Lead ", "Principal ",
]
SENIORITY_SUFFIXES = [
    "主管", "负责人", "总监", "VP", "专家", "专员", "助理", "leader",
]
DOMAIN_PREFIXES = [
    "AI", "大模型", "B端", "C端", "硬件", "数据", "金融", "出海",
    "电商", "游戏", "SaaS", "医疗", "教育", "自动驾驶", "Web3",
    "商业化", "增长", "策略", "招聘",
]
COLLOQUIAL_TEMPLATES = [
    "做{x}的", "搞{x}的", "负责{x}",
    "做{x}的哥们", "搞{x}的同事",
    "我们公司做{x}", "招一个做{x}的",
    "{x}方向", "{x}岗位", "{x}这块",
    "类似{x}这种的",
]
CONTEXT_TEMPLATES = [
    "{x}，在字节做推荐",
    "{x}，在腾讯带 10 人团队",
    "{x}，在阿里做 B 端",
    "{x}，负责 AI 产品方向",
    "{x}，汇报给 CEO",
    "{x}，上一段在美团",
    "{x}，之前做过开发",
    "{x}，偏管理岗",
    "{x}，IC 路线",
    "{x}，跨部门协作多",
    "{x}，年包 60-80w",
    "{x}，大厂背景",
    "{x}，带过 20+ 人团队",
    "{x}，刚毕业 3 年",
]
EN_ZH_MIX = {
    "产品经理": ["PM", "product manager", "Product Manager", "prod mgr"],
    "项目经理": ["PMO", "project manager", "Project Mgr"],
    "软件工程师": ["SDE", "software engineer", "dev", "coder", "Developer"],
    "算法工程师": ["ML engineer", "MLE", "algo engineer", "算法同学"],
    "数据分析师": ["DA", "data analyst", "BI"],
    "测试工程师": ["QA", "test engineer", "SDET"],
    "设计师": ["designer", "UI/UX", "UX designer"],
    "运营经理": ["operations", "运营同学", "OP"],
    "市场经理": ["marketing manager", "MKT", "marketing"],
    "销售经理": ["sales", "sales manager", "BD"],
    "人力资源经理": ["HR", "HRBP", "HR manager"],
    "财务经理": ["finance manager", "FP&A", "财务同学"],
    "投资经理": ["VP investment", "投资总监", "IR"],
    "法务经理": ["legal", "legal counsel"],
    "供应链经理": ["SCM", "supply chain"],
    "架构师": ["architect", "tech lead", "TL"],
}
TYPO_RULES = [
    (r"经理", "经里"),
    (r"工程师", "工程"),
    (r"产品", "产吕"),
    (r"运营", "运营 "),
    (r"总监", "总监长"),
]


# ═══════════════════════════════════════════════════════════
#  工具
# ═══════════════════════════════════════════════════════════

def _load_kg() -> dict:
    return json.loads(KG_PATH.read_text(encoding="utf-8"))


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _pick_hard_negative(
    positive_id: str,
    category: str,
    category_to_variants: dict[str, list[tuple[str, str, str]]],
    std_to_variants: dict[str, list[tuple[str, str, str]]],
    std_label: dict[str, str],
    cross_refs: list[dict],
    rng: random.Random,
) -> tuple[str, str] | None:
    """多级回退挑困难负样本。

    1) 同类别 + 不同标准
    2) cross_references 里与 positive_id 相关的标准（协作/同族/晋升）
    3) 含相同后缀（经理/工程师/总监/师）的其他标准
    4) 随机跨类别
    """
    pool = [
        (v, sid, label)
        for v, sid, label in category_to_variants.get(category, [])
        if sid != positive_id
    ]
    if pool:
        v, sid, _ = rng.choice(pool)
        return v, sid

    related_ids: list[str] = []
    for cr in cross_refs:
        if cr.get("from") == positive_id and cr.get("to") != positive_id:
            related_ids.append(cr["to"])
        elif cr.get("to") == positive_id and cr.get("from") != positive_id:
            related_ids.append(cr["from"])
    related_ids = [sid for sid in related_ids if sid in std_to_variants]
    if related_ids:
        sid = rng.choice(related_ids)
        v, sid2, _ = rng.choice(std_to_variants[sid])
        return v, sid2

    pos_label = std_label.get(positive_id, "")
    suffixes = ["经理", "工程师", "总监", "师", "分析师", "助理", "专员"]
    for sfx in suffixes:
        if sfx in pos_label:
            cand_ids = [
                sid for sid, lbl in std_label.items()
                if sid != positive_id and sfx in lbl
            ]
            if cand_ids:
                sid = rng.choice(cand_ids)
                v, sid2, _ = rng.choice(std_to_variants[sid])
                return v, sid2
            break

    other_ids = [sid for sid in std_to_variants if sid != positive_id]
    if not other_ids:
        return None
    sid = rng.choice(other_ids)
    v, sid2, _ = rng.choice(std_to_variants[sid])
    return v, sid2


def _pick_random_negative(
    positive_id: str,
    category: str,
    category_to_variants: dict[str, list[tuple[str, str, str]]],
    rng: random.Random,
) -> tuple[str, str] | None:
    """跨类别随机负样本。"""
    other_cats = [c for c in category_to_variants.keys() if c != category]
    if not other_cats:
        return None
    c = rng.choice(other_cats)
    v, sid, _ = rng.choice(category_to_variants[c])
    return v, sid


# ═══════════════════════════════════════════════════════════
#  模板生成
# ═══════════════════════════════════════════════════════════

def gen_t1_raw(variants_by_std: dict[str, list[dict]]) -> list[dict]:
    out = []
    for sid, vs in variants_by_std.items():
        for v in vs:
            out.append({
                "anchor": v["name"],
                "positive_id": sid,
                "positive": v["std_label"],
                "category": v["category"],
                "generation_type": "T1_raw",
            })
    return out


def gen_t2_seniority(
    variants_by_std: dict[str, list[dict]], rng: random.Random, per_std: int = 12
) -> list[dict]:
    out = []
    for sid, vs in variants_by_std.items():
        label = vs[0]["std_label"]
        cat = vs[0]["category"]
        cores = list({v["name"] for v in vs if v["type"] in ("canonical", "variant")})
        if not cores:
            continue
        for _ in range(per_std):
            base = rng.choice(cores)
            pfx = rng.choice(SENIORITY_PREFIXES)
            if rng.random() < 0.4:
                sfx = rng.choice(SENIORITY_SUFFIXES)
                anchor = f"{pfx}{base}{sfx}"
            else:
                anchor = f"{pfx}{base}"
            out.append({
                "anchor": _normalize(anchor),
                "positive_id": sid,
                "positive": label,
                "category": cat,
                "generation_type": "T2_seniority",
            })
    return out


def gen_t3_domain(
    variants_by_std: dict[str, list[dict]], rng: random.Random, per_std: int = 12
) -> list[dict]:
    out = []
    for sid, vs in variants_by_std.items():
        label = vs[0]["std_label"]
        cat = vs[0]["category"]
        cores = list({v["name"] for v in vs if v["type"] in ("canonical", "variant")})
        if not cores:
            continue
        for _ in range(per_std):
            base = rng.choice(cores)
            dom = rng.choice(DOMAIN_PREFIXES)
            anchor = f"{dom}{base}"
            out.append({
                "anchor": _normalize(anchor),
                "positive_id": sid,
                "positive": label,
                "category": cat,
                "generation_type": "T3_domain",
            })
    return out


def gen_t4_colloquial(
    variants_by_std: dict[str, list[dict]], rng: random.Random, per_std: int = 12
) -> list[dict]:
    out = []
    for sid, vs in variants_by_std.items():
        label = vs[0]["std_label"]
        cat = vs[0]["category"]
        names = [v["name"] for v in vs]
        for _ in range(per_std):
            base = rng.choice(names)
            tpl = rng.choice(COLLOQUIAL_TEMPLATES)
            anchor = tpl.format(x=base)
            out.append({
                "anchor": _normalize(anchor),
                "positive_id": sid,
                "positive": label,
                "category": cat,
                "generation_type": "T4_colloquial",
            })
    return out


def gen_t5_en_zh_mix(
    variants_by_std: dict[str, list[dict]], rng: random.Random, per_std: int = 8
) -> list[dict]:
    out = []
    for sid, vs in variants_by_std.items():
        label = vs[0]["std_label"]
        cat = vs[0]["category"]
        en_variants = EN_ZH_MIX.get(label, [])
        if not en_variants:
            continue
        for _ in range(per_std):
            en = rng.choice(en_variants)
            style = rng.random()
            if style < 0.3:
                anchor = en.upper()
            elif style < 0.55:
                anchor = en.lower()
            elif style < 0.8:
                anchor = f"{en} 方向"
            else:
                anchor = f"高级{en}" if rng.random() < 0.5 else f"{en}(senior)"
            out.append({
                "anchor": _normalize(anchor),
                "positive_id": sid,
                "positive": label,
                "category": cat,
                "generation_type": "T5_en_zh_mix",
            })
    return out


def gen_t6_typo(
    variants_by_std: dict[str, list[dict]], rng: random.Random, per_std: int = 6
) -> list[dict]:
    out = []
    for sid, vs in variants_by_std.items():
        label = vs[0]["std_label"]
        cat = vs[0]["category"]
        names = [v["name"] for v in vs]
        for _ in range(per_std):
            base = rng.choice(names)
            rule = rng.choice(TYPO_RULES)
            anchor = re.sub(rule[0], rule[1], base, count=1)
            if anchor == base and len(base) > 1:
                i = rng.randint(0, len(base) - 2)
                anchor = base[:i] + base[i + 1] + base[i] + base[i + 2:]
            out.append({
                "anchor": _normalize(anchor),
                "positive_id": sid,
                "positive": label,
                "category": cat,
                "generation_type": "T6_typo",
            })
    return out


def gen_t7_context(
    variants_by_std: dict[str, list[dict]], rng: random.Random, per_std: int = 10
) -> list[dict]:
    out = []
    for sid, vs in variants_by_std.items():
        label = vs[0]["std_label"]
        cat = vs[0]["category"]
        names = [v["name"] for v in vs]
        for _ in range(per_std):
            base = rng.choice(names)
            tpl = rng.choice(CONTEXT_TEMPLATES)
            anchor = tpl.format(x=base)
            out.append({
                "anchor": _normalize(anchor),
                "positive_id": sid,
                "positive": label,
                "category": cat,
                "generation_type": "T7_context",
            })
    return out


# ═══════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════

def build_variant_pool(kg: dict) -> tuple[dict, dict, dict, dict]:
    variants_by_std: dict[str, list[dict]] = defaultdict(list)
    category_to_variants: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    std_to_variants: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    std_label: dict[str, str] = {}

    for st in kg["standard_titles"]:
        sid, label, cat = st["id"], st["label"], st["category"]
        std_label[sid] = label

        variants_by_std[sid].append({
            "name": label, "type": "canonical", "std_label": label, "category": cat,
        })
        category_to_variants[cat].append((label, sid, label))
        std_to_variants[sid].append((label, sid, label))

        for v in st.get("variants", []) or []:
            variants_by_std[sid].append({
                "name": v, "type": "variant", "std_label": label, "category": cat,
            })
            category_to_variants[cat].append((v, sid, label))
            std_to_variants[sid].append((v, sid, label))

        for v in st.get("senior_variants", []) or []:
            variants_by_std[sid].append({
                "name": v, "type": "senior_variant", "std_label": label, "category": cat,
            })
            category_to_variants[cat].append((v, sid, label))
            std_to_variants[sid].append((v, sid, label))

        for _dom, vs in (st.get("domain_variants", {}) or {}).items():
            for v in vs:
                variants_by_std[sid].append({
                    "name": v, "type": "domain_variant", "std_label": label, "category": cat,
                })
                category_to_variants[cat].append((v, sid, label))
                std_to_variants[sid].append((v, sid, label))

    return variants_by_std, category_to_variants, std_to_variants, std_label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=10000, help="目标样本量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    kg = _load_kg()
    variants_by_std, category_to_variants, std_to_variants, std_label = build_variant_pool(kg)
    cross_refs = kg.get("cross_references", [])

    num_std = len(variants_by_std)
    remaining = args.target
    per_std_base = max(1, (args.target - 629) // (num_std * 6))

    samples: list[dict] = []
    samples += gen_t1_raw(variants_by_std)
    samples += gen_t2_seniority(variants_by_std, rng, per_std=per_std_base + 4)
    samples += gen_t3_domain(variants_by_std, rng, per_std=per_std_base + 4)
    samples += gen_t4_colloquial(variants_by_std, rng, per_std=per_std_base + 4)
    samples += gen_t5_en_zh_mix(variants_by_std, rng, per_std=per_std_base + 2)
    samples += gen_t6_typo(variants_by_std, rng, per_std=per_std_base)
    samples += gen_t7_context(variants_by_std, rng, per_std=per_std_base + 3)

    seen_anchors = set()
    deduped: list[dict] = []
    for s in samples:
        key = (s["anchor"], s["positive_id"])
        if key in seen_anchors:
            continue
        seen_anchors.add(key)
        deduped.append(s)

    if len(deduped) < args.target:
        print(f"[WARN] 去重后仅 {len(deduped)} 条，未达到 {args.target}，将追加口语化+上下文样本")
        extra_needed = args.target - len(deduped)
        extra = gen_t4_colloquial(variants_by_std, rng, per_std=extra_needed // num_std + 2) + \
                gen_t7_context(variants_by_std, rng, per_std=extra_needed // num_std + 2)
        for s in extra:
            key = (s["anchor"], s["positive_id"])
            if key not in seen_anchors:
                seen_anchors.add(key)
                deduped.append(s)
                if len(deduped) >= args.target:
                    break

    rng.shuffle(deduped)
    deduped = deduped[: args.target]
    print(f"[INFO] 生成 {len(deduped)} 条 anchor 样本")

    records: list[dict] = []
    for i, s in enumerate(deduped, 1):
        hn = _pick_hard_negative(
            s["positive_id"], s["category"],
            category_to_variants, std_to_variants, std_label, cross_refs, rng,
        )
        rn = _pick_random_negative(s["positive_id"], s["category"], category_to_variants, rng)
        records.append({
            "pair_id": f"P{i:06d}",
            "anchor": s["anchor"],
            "positive": s["positive"],
            "positive_id": s["positive_id"],
            "hard_negative": hn[0] if hn else "",
            "hard_negative_id": hn[1] if hn else "",
            "random_negative": rn[0] if rn else "",
            "random_negative_id": rn[1] if rn else "",
            "category": s["category"],
            "generation_type": s["generation_type"],
        })

    cut = int(len(records) * (1 - args.eval_ratio))
    train_rows = records[:cut]
    eval_rows = records[cut:]

    train_csv = OUT_DIR / "embedding_train.csv"
    eval_csv = OUT_DIR / "embedding_eval.csv"
    train_jsonl = OUT_DIR / "embedding_train.jsonl"

    fields = list(records[0].keys())
    for path, rows in [(train_csv, train_rows), (eval_csv, eval_rows)]:
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"[OK] {path.relative_to(ROOT)}  ({len(rows)} rows)")

    with train_jsonl.open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps({
                "query": r["anchor"],
                "pos": [r["positive"]],
                "neg": [r["hard_negative"], r["random_negative"]],
                "meta": {
                    "positive_id": r["positive_id"],
                    "category": r["category"],
                    "generation_type": r["generation_type"],
                },
            }, ensure_ascii=False) + "\n")
    print(f"[OK] {train_jsonl.relative_to(ROOT)}  ({len(train_rows)} lines)")

    stats = defaultdict(int)
    cat_stats = defaultdict(int)
    for r in records:
        stats[r["generation_type"]] += 1
        cat_stats[r["category"]] += 1
    print("\n[STATS] 生成模板分布:")
    for k in sorted(stats):
        print(f"  {k:>16}: {stats[k]:>5}  ({stats[k]/len(records)*100:.1f}%)")
    print("\n[STATS] 类别分布:")
    for k in sorted(cat_stats, key=lambda x: -cat_stats[x]):
        print(f"  {k:>8}: {cat_stats[k]:>5}")

    print("\n[SAMPLE] 前 5 条记录:")
    for r in records[:5]:
        print(f"  {r['pair_id']}  anchor='{r['anchor']}'  pos='{r['positive']}'  "
              f"hard_neg='{r['hard_negative']}'  ({r['generation_type']})")

    print(f"\n[DONE] 共生成 {len(records)} 条训练数据 (train={len(train_rows)}, eval={len(eval_rows)})")


if __name__ == "__main__":
    main()
