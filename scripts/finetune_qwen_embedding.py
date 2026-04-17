# -*- coding: utf-8 -*-
"""
Qwen3-Embedding-0.6B 最小 LoRA 微调脚本。

数据：  dataset/embedding_train.csv
输出：  models/Qwen3-Embedding-0.6B-ft/   (LoRA 合并后的完整模型，可直接 SentenceTransformer 加载)
Loss :  MultipleNegativesRankingLoss (anchor / positive / hard_negative)
LoRA :  r=8, alpha=16, target=["q_proj","k_proj","v_proj","o_proj"], bias=none

─────────────────────────────────────────────────────────────
 ⚠ 运行时长预期 ⚠
─────────────────────────────────────────────────────────────
  GPU (3060/3090/A10/T4 都可):
      默认参数 ~1000 样本 + batch 16，约 3-8 min 完成一个 epoch
      推荐: python scripts/finetune_qwen_embedding.py --samples 0 --epochs 1 --batch 16
      (全量 9000 条 ~15-30 min)

  CPU only:
      0.6B 模型在 CPU 上 forward+backward 极慢（实测 ~60s/step）
      强烈建议换到 GPU 机器跑；如一定要在 CPU 上演示：
          python scripts/finetune_qwen_embedding.py --samples 32 --batch 2 --max-len 32
      约 15-30 min 完成，纯为流程验证

  无论哪种环境，sanity eval (开始前/结束后各 1 次) 都能快速看到
  pos/hard_neg 相似度差距，5 秒即出结果，可先行观察 base 能力。
─────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from sentence_transformers import InputExample, SentenceTransformer, losses  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


BASE_MODEL_DIR = ROOT / "models" / "Qwen3-Embedding-0.6B"
FT_MODEL_DIR = ROOT / "models" / "Qwen3-Embedding-0.6B-ft"
TRAIN_CSV = ROOT / "dataset" / "embedding_train.csv"


def load_triplets(path: Path, max_samples: int, seed: int) -> list[InputExample]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("anchor") and r.get("positive") and r.get("hard_negative"):
                rows.append(r)
    if max_samples > 0 and len(rows) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:max_samples]
    return [
        InputExample(texts=[r["anchor"], r["positive"], r["hard_negative"]])
        for r in rows
    ]


def apply_lora(model: SentenceTransformer, r: int, alpha: int, dropout: float) -> None:
    """给第一层 Transformer 模块注入 LoRA adapter。"""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise RuntimeError("缺少 peft，请先 pip install peft") from e

    first = model._first_module()
    transformer = first.auto_model

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="FEATURE_EXTRACTION",
    )
    peft_model = get_peft_model(transformer, config)
    first.auto_model = peft_model
    peft_model.print_trainable_parameters()


def merge_and_save(model: SentenceTransformer, out_dir: Path) -> None:
    """把 LoRA 权重合并回 base model，再保存为独立 SentenceTransformer 目录。"""
    first = model._first_module()
    peft_model = first.auto_model
    if hasattr(peft_model, "merge_and_unload"):
        merged = peft_model.merge_and_unload()
        first.auto_model = merged
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))


def quick_sanity_eval(model: SentenceTransformer, samples: int = 50) -> None:
    import numpy as np

    rows = []
    with TRAIN_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["hard_negative"]:
                rows.append(r)
    rng = random.Random(0)
    rng.shuffle(rows)
    rows = rows[:samples]

    anchors = [r["anchor"] for r in rows]
    pos = [r["positive"] for r in rows]
    neg = [r["hard_negative"] for r in rows]
    with torch.no_grad():
        a = model.encode(anchors, normalize_embeddings=True, convert_to_numpy=True)
        p = model.encode(pos, normalize_embeddings=True, convert_to_numpy=True)
        n = model.encode(neg, normalize_embeddings=True, convert_to_numpy=True)
    sim_pos = (a * p).sum(axis=1)
    sim_neg = (a * n).sum(axis=1)
    acc = float((sim_pos > sim_neg).mean())
    print(f"[SANITY] triplet accuracy (pos > hard_neg) = {acc:.4f}  "
          f"(mean_pos={sim_pos.mean():.3f}, mean_neg={sim_neg.mean():.3f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000,
                        help="训练样本数，0=全量；CPU 演示建议 32")
    parser.add_argument("--batch", type=int, default=16,
                        help="GPU 建议 16-32；CPU 建议 2-4")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=64,
                        help="序列最大长度；职称都很短，GPU 64 / CPU 32 足矣")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(FT_MODEL_DIR))
    parser.add_argument("--base", type=str, default=str(BASE_MODEL_DIR))
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[FT] device = {device}")
    print(f"[FT] base model = {args.base}")
    print(f"[FT] output     = {args.output}")

    t0 = time.time()
    examples = load_triplets(TRAIN_CSV, args.samples, args.seed)
    print(f"[FT] 训练三元组数: {len(examples)}")
    print(f"[FT] 样例: '{examples[0].texts[0]}' -> '{examples[0].texts[1]}'  "
          f"[neg='{examples[0].texts[2]}']")

    model = SentenceTransformer(args.base, device=device)
    model.max_seq_length = args.max_len
    print(f"[FT] max_seq_length = {model.max_seq_length}")

    print("\n[FT] 微调前 sanity check...")
    quick_sanity_eval(model, samples=50)

    print("\n[FT] 注入 LoRA adapter...")
    apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)

    loader = DataLoader(examples, batch_size=args.batch, shuffle=True)
    loss = losses.MultipleNegativesRankingLoss(model)

    steps_per_epoch = math.ceil(len(examples) / args.batch)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    print(f"[FT] steps/epoch={steps_per_epoch}, total={total_steps}, warmup={warmup}")

    print("\n[FT] 开始训练...")
    t_train = time.time()
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
        use_amp=(device == "cuda"),
    )
    print(f"[FT] 训练耗时: {time.time()-t_train:.1f}s")

    print("\n[FT] 合并 LoRA 权重并保存...")
    merge_and_save(model, Path(args.output))
    print(f"[FT] 已保存到: {args.output}")

    print("\n[FT] 微调后 sanity check (重新加载合并后的模型)...")
    merged = SentenceTransformer(args.output, device=device)
    quick_sanity_eval(merged, samples=50)

    print(f"\n[DONE] 总耗时 {time.time()-t0:.1f}s")
    print(f"下一步可以跑:\n"
          f"    python scripts/compare_ft.py --sample 500\n"
          f"来对比 '未微调 vs 微调后' 的端到端准确率。")


if __name__ == "__main__":
    main()
