# CSV 测试集 & 知识图谱（agent-friendly 版）

> 由 `scripts/convert_to_csv.py` 从 `test_cases_v2.json` / `knowledge_graph_v2.json` 自动生成。
> 文件编码统一为 **UTF-8 with BOM**，Excel / pandas / polars / duckdb 直接可读。

## 文件清单

| 文件 | 说明 | 主要字段 |
| --- | --- | --- |
| `queries.csv` | 测试问题主表（1500 条，覆盖 L1/L2/L3） | `test_id`, `input_title`, `context`, `difficulty`, `target_level`, `expected_standard_title`, `expected_standard_id`, `expected_seniority`, `expected_domain` |
| `kg_standards.csv` | 标准职称表（19 条） | `standard_id`, `label`, `category`, `related_skills`（`\|` 分隔） |
| `kg_variants.csv` | 变体 -> 标准 的映射边（629 条） | `variant_id`, `variant_name`, `standard_id`, `standard_label`, `variant_type`（canonical / variant / senior\_variant / domain\_variant）, `domain`, `seniority_hint` |
| `kg_cross_refs.csv` | 标准职称之间的交叉引用（10 条） | `from_id`, `to_id`, `type`, `description` |
| `kg_dimensions.csv` | 维度枚举（seniority / domain / scope / tech\_stack） | `dimension_key`, `dimension_label`, `value` |
| `eval_report.csv` | *（运行评测后生成）* 每条 query 的 A/B 对比结果 | 见 `scripts/eval_embedding.py` |
| `embedding_train.csv` | **9000** 条 embedding 微调训练三元组 | `pair_id`, `anchor`, `positive`, `positive_id`, `hard_negative`, `hard_negative_id`, `random_negative`, `random_negative_id`, `category`, `generation_type` |
| `embedding_eval.csv` | **1000** 条对应验证集 | 字段同上 |
| `embedding_train.jsonl` | 训练集 JSONL 版本（HF datasets / sentence-transformers 直接兼容） | `{query, pos:[...], neg:[...], meta}` |

## Embedding 训练数据生成规则

运行 `python scripts/generate_training_data.py --target 10000` 生成共 **10000** 条 (train 9000 / eval 1000)。

### 7 类生成模板

| 模板 | 占比 | 说明 | 示例 |
| --- | --- | --- | --- |
| `T1_raw` | 6.2% | 原始 variant 直接作为 anchor | `产品owner` → 产品经理 |
| `T2_seniority` | 13.6% | 叠加职级前缀 | `Senior Sales Manager` → 销售经理 |
| `T3_domain` | 12.7% | 叠加行业/领域前缀 | `AI产品经理` → 产品经理 |
| `T4_colloquial` | 29.7% | 口语化包装 | `做用户增长这种的` / `搞实施经理的` |
| `T5_en_zh_mix` | 2.3% | 中英混写、大小写扰动 | `PM 方向` / `HRBP` |
| `T6_typo` | 11.8% | 简拼/错字/字序颠倒（真实脏输入） | `品产经理` / `资深设师计` |
| `T7_context` | 23.7% | variant + 真实上下文 | `后端 leader，跨部门协作多` |

### 负样本生成（4 级回退）

1. **同类别 + 不同标准**（真正的困难负样本）
2. **cross_references 相关标准**（协作/同族/晋升）
3. **同后缀匹配**（都含"经理"/"工程师"/"总监"/"师"）
4. **随机跨类别**

最终 `hard_negative` 列 **100% 覆盖率**（无空值）。

### 可直接用于以下训练框架

- `sentence-transformers` 的 `MultipleNegativesRankingLoss` / `TripletLoss`
- HuggingFace `datasets` + `trl`
- 任何需要 (anchor, positive, negative) 三元组的 contrastive learning 流程

示例加载：
```python
import pandas as pd
df = pd.read_csv("dataset/embedding_train.csv")
triplets = list(zip(df["anchor"], df["positive"], df["hard_negative"]))
```

## LoRA 微调 + 端到端对比（完整 pipeline）

### 1) 微调（推荐 GPU 单卡）

```powershell
# GPU 推荐：全量 9000 样本，1 epoch，约 15-30 min
python scripts/finetune_qwen_embedding.py --samples 0 --epochs 1 --batch 16

# GPU 快跑 1000 样本：~3-8 min
python scripts/finetune_qwen_embedding.py --samples 1000 --batch 16

# CPU 演示（不推荐做真实训练，0.6B 模型在 CPU 上极慢）
python scripts/finetune_qwen_embedding.py --samples 32 --batch 2 --max-len 32
```

产出：`models/Qwen3-Embedding-0.6B-ft/`（已 merge LoRA 权重的完整 SentenceTransformer 目录）

### 2) 一键对比 "base vs ft"

```powershell
# 全量 1500 query
python scripts/compare_ft.py

# 小样本快速看数
python scripts/compare_ft.py --sample 300

# 只跑 base（微调还没做）
python scripts/compare_ft.py --base-only
```

产出：
- `dataset/eval_report_base.csv` — 未微调每条 query 的详细结果
- `dataset/eval_report_ft.csv` — 微调后每条 query 的详细结果
- `dataset/compare_ft_summary.csv` — 关键指标对比（总体 acc、L3 救回、按难度/按 target_level）

控制台会直接打印：

```
[COMPARE]  未微调 base  vs  LoRA 微调后 ft
========================================================================
  total queries                     1500           1500
  pure KG acc                     0.8xxx         0.8xxx
  KG+Embedding acc                0.9xxx         0.9xxx
  Embedding 救回                  +xx            +xx
  Embedding 回退错                -xx            -xx
  net gain                       +xx            +xx
  -> 微调带来 KG+Embedding 准确率提升: +0.0xxx (+x.xx 个百分点)
```

### 3) 运行时 sanity check（5s 即出）

微调脚本里包含 `quick_sanity_eval()`，会在训练前后各跑 50 条三元组，快速输出：

```
[SANITY] triplet accuracy (pos > hard_neg) = 0.8600
         (mean_pos=0.664, mean_neg=0.559)
```

**基准值**：Qwen3-Embedding-0.6B 未微调在我们任务上 **triplet accuracy ≈ 0.86**，说明已有较强 zero-shot 能力，微调目标是把剩余 14% 的难例拉回来。

## 混合 Pipeline（KG 规则 + LLM 泛化）

这是最终上线架构，也是领导确认的方案：

```
输入 → L1 精确 → L2 高置信度模糊 → L3 LLM 泛化（把 KG 词表作 prompt 上下文）
        │ 命中    │ 命中              │ 命中 → 输出
        │         │                   │
        └─────────┴────── 全部 miss ─┴─→ 返回 L0_miss
```

### 核心文件

| 文件 | 作用 |
| --- | --- |
| `prompt_builder.py` | 构建 KG 上下文 prompt（19 个标准 + 维度 + 交叉引用） + JSON 输出约束 |
| `llm_engine.py` | 3 种可插拔 LLM 后端：`embedding` / `local` / `api` |
| `pipeline.py` | `HybridPipeline` 主调度类 |
| `scripts/eval_pipeline.py` | 端到端评测（KG + LLM） |
| `scripts/analyze_llm_failures.py` | 聚合 LLM 失败 case，输出词表扩展候选 |
| `scripts/download_qwen_llm.py` | 下载 Qwen3-0.6B 生成模型（用于 local 后端） |

### 运行方式

```powershell
# A 先看反例池（最便宜，不调 LLM）
python scripts/eval_pipeline.py --no-llm --sample 300

# B Embedding 后端泛化（已部署，推荐默认）
python scripts/eval_pipeline.py --backend embedding --sample 300

# C 本地 Qwen3-0.6B 生成模型（先下载）
python scripts/download_qwen_llm.py
python scripts/eval_pipeline.py --backend local --sample 50

# D OpenAI 兼容 API（最快出强效果，如 DashScope）
$env:LLM_API_KEY='sk-xxx'
$env:LLM_API_BASE='https://dashscope.aliyuncs.com/compatible-mode/v1'
$env:LLM_MODEL='qwen3-0.6b'
python scripts/eval_pipeline.py --backend api --sample 300

# E 聚合 LLM 失败 case，为下一轮 "LLM 优化词表" 准备输入
python scripts/analyze_llm_failures.py
```

### 实测结果（300 条 query，embedding 后端）

| 指标 | 值 |
| --- | --- |
| 纯 KG (L1+L2, threshold=0.75) 解决率 | 49.0% |
| KG 命中部分准确率 | 100.0% |
| LLM 接手的反例数 | 153 |
| **LLM 救回正确数** | **127 (83.0%)** |
| **整条 pipeline 最终准确率** | **91.33%** |
| **LLM 带来的绝对提升** | **+42.33 个百分点** |

按难度：
- L1 97.9% / L2 100% / L3 86.7% / out_of_graph 71.2%

### LLM 失败的 case 分析（领导要的"词表优化线索"）

运行 `scripts/analyze_llm_failures.py` 得到三份输出：

- `dataset/llm_failures_by_std.csv` — 按期望 standard 聚合，显示 KG 盲区
- `dataset/llm_failures_raw.csv` — 全部失败原始样本
- `dataset/kg_expand_candidates.csv` — 直接喂给下一轮 LLM 做词表优化的输入

示例发现：
- `product_manager` 盲区最大（5 条）：`用户研究` / `竞品分析` / `需求分析师` 等
- `general_manager` 盲区（4 条）：`战略规划总监` / `数字化转型负责人`
- `software_engineer` 盲区（4 条）：`RPA开发工程师` / `技术合伙人`
- `marketing_manager` 盲区（3 条）：`GTM策略` / `CMO（偏策略）`

这些就是下一轮词表扩展的靶子。

## 使用建议

### pandas
```python
import pandas as pd
queries = pd.read_csv("dataset/queries.csv")
variants = pd.read_csv("dataset/kg_variants.csv")
```

### DuckDB（推荐，零拷贝）
```sql
SELECT difficulty, COUNT(*)
FROM read_csv_auto('dataset/queries.csv')
GROUP BY difficulty;
```

### 重新生成
```powershell
python scripts/convert_to_csv.py
```

## 字段约定

- `difficulty` ∈ {`easy`, `medium`, `hard`}
- `target_level` ∈ {`L1`, `L2`, `L3`}
    - `L1` → 精确匹配（exact）
    - `L2` → 模糊匹配（fuzzy / embedding）
    - `L3` → 图谱推理 / LLM
- `variant_type`：
    - `canonical` — 标准职称本身
    - `variant` — 通用同义
    - `senior_variant` — 含职级的同义（VP / 总监 / CPO 等）
    - `domain_variant` — 行业/领域特化（AI产品经理、B端产品经理等）
- `expected_seniority` / `expected_domain` 为空代表不作强制要求
