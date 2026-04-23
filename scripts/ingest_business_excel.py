# -*- coding: utf-8 -*-
"""
将研发提供的 talent 业务导出（.xlsx）导入到 dataset/business/，并生成聚合摘要供 API / 演示使用。

用法（路径按本机调整）：
  python scripts/ingest_business_excel.py \\
    --skills "C:/Users/.../skill_na_....xlsx" \\
    --titles "C:/Users/.../title_....xlsx"

依赖：pandas, openpyxl（见 requirements.txt）
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "dataset" / "business"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _norm_series(s):
    return (
        s.astype(str)
        .str.strip()
        .replace({"nan": ""})
    )


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", type=Path, help="技能导出 xlsx（key_name, skill_name）")
    ap.add_argument("--titles", type=Path, help="职称导出 xlsx（title）")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "version": 1,
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {},
    }

    if args.skills and args.skills.exists():
        df = pd.read_excel(args.skills)
        df.columns = [str(c).strip() for c in df.columns]
        if "key_name" not in df.columns or "skill_name" not in df.columns:
            raise SystemExit(f"技能表缺少列: {df.columns.tolist()}")
        df["skill_name"] = _norm_series(df["skill_name"])
        df["key_name"] = _norm_series(df["key_name"])
        df = df[(df["skill_name"] != "") & (df["key_name"] != "")]
        skills_csv = OUT_DIR / "talent_skills_full.csv"
        df.to_csv(skills_csv, index=False, encoding="utf-8-sig")
        by_key = (
            df.groupby("key_name")["skill_name"]
            .agg(lambda s: sorted(set(s))[:12])
            .reset_index(name="sample_skills")
        )
        key_counts = df.groupby("key_name").size().reset_index(name="row_count").sort_values(
            "row_count", ascending=False
        )
        summary = key_counts.merge(by_key, on="key_name", how="left")
        summary_path = OUT_DIR / "talent_skills_by_key_summary.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        manifest["outputs"]["talent_skills_full"] = skills_csv.relative_to(ROOT).as_posix()
        manifest["outputs"]["talent_skills_by_key_summary"] = summary_path.relative_to(ROOT).as_posix()
        manifest["skills_source"] = {
            "path": str(args.skills.resolve()),
            "sha256_16": _sha256(args.skills),
            "rows_kept": int(len(df)),
            "distinct_keys": int(df["key_name"].nunique()),
            "distinct_skill_texts": int(df["skill_name"].nunique()),
        }

    if args.titles and args.titles.exists():
        df = pd.read_excel(args.titles)
        df.columns = [str(c).strip() for c in df.columns]
        if "title" not in df.columns:
            raise SystemExit(f"职称表缺少 title 列: {df.columns.tolist()}")
        df["title"] = _norm_series(df["title"])
        df = df[df["title"] != ""]
        titles_csv = OUT_DIR / "talent_titles_full.csv"
        df.to_csv(titles_csv, index=False, encoding="utf-8-sig")
        freq = df["title"].value_counts().reset_index()
        freq.columns = ["title", "count"]
        top_path = OUT_DIR / "talent_titles_top_freq.csv"
        freq.head(8000).to_csv(top_path, index=False, encoding="utf-8-sig")
        manifest["outputs"]["talent_titles_full"] = titles_csv.relative_to(ROOT).as_posix()
        manifest["outputs"]["talent_titles_top_freq"] = top_path.relative_to(ROOT).as_posix()
        manifest["titles_source"] = {
            "path": str(args.titles.resolve()),
            "sha256_16": _sha256(args.titles),
            "rows_kept": int(len(df)),
            "distinct_titles": int(df["title"].nunique()),
        }

    mp = OUT_DIR / "business_manifest.json"
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(mp)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
