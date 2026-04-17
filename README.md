# 职位标准化系统（KG + Embedding + LLM）

这是一个用于**职位名称标准化**的项目：把用户输入的自由职位描述（口语化、错字、中英混写、上下文混杂）映射到统一的标准职位体系中。

项目采用分层架构：

- **L1**：知识图谱精确匹配（高精度、低成本）
- **L2**：规则模糊匹配 / embedding 召回（提升覆盖）
- **L3**：LLM 泛化推理（解决 L1/L2 无法覆盖的复杂反例）

---

## 1. 这个项目解决什么问题

在招聘、简历解析、人才库检索场景里，同一岗位会有大量写法：

- 同义：`产品经理` / `PM` / `Product Manager`
- 职级：`高级产品经理` / `产品总监` / `CPO`
- 领域：`AI产品经理` / `B端产品经理` / `增长产品经理`
- 脏输入：`品产经理`、`做用户增长这种的`、`搞K8s的后端大哥`

如果不做标准化，就很难做精确检索、统计分析、推荐匹配。  
本项目目标是输出统一结构：

- `standard_id`
- `standard_title`
- `seniority`
- `domain`
- `confidence`

---

## 2. 项目结构总览

```text
zhishitupu-demo/
├── app.py                        # FastAPI 服务（KG 基础能力）
├── pipeline.py                   # 混合调度：L1/L2 + L3 LLM
├── llm_engine.py                 # L3 引擎：embedding/local/api 三种后端
├── prompt_builder.py             # 组装 KG 上下文 prompt + JSON 解析
├── embedding_service.py          # Qwen3-Embedding 检索服务与索引
├── knowledge_graph_v2.json       # 原始知识图谱
├── test_cases_v2.json            # 原始测试集（JSON）
├── dataset/                      # CSV 数据资产（训练/评测/分析）
├── scripts/                      # 生成、训练、评测、分析脚本
├── models/                       # 本地模型（embedding / llm）
└── requirements.txt              # Python 依赖
```

---

## 3. 数据文件怎么理解（最重要）

### 3.1 原始数据

- `knowledge_graph_v2.json`  
  职位知识图谱主文件，包含：
  - 标准职位（`standard_titles`）
  - 变体（`variants` / `senior_variants` / `domain_variants`）
  - 维度枚举（`dimensions`）
  - 跨职位关系（`cross_references`）

- `test_cases_v2.json`  
  原始测试题库（JSON 格式），包含输入标题、上下文、期望输出。

### 3.2 agent-friendly CSV（推荐日常使用）

在 `dataset/` 下，核心文件如下：

- `queries.csv`：主测试集（1500 条）
- `kg_standards.csv`：标准职位表（19 条）
- `kg_variants.csv`：词表映射（629 条）
- `kg_cross_refs.csv`：标准之间关系
- `kg_dimensions.csv`：维度枚举

### 3.3 训练与评测产物

- `embedding_train.csv` / `embedding_eval.csv`  
  embedding 微调三元组（anchor/positive/hard_negative）
- `embedding_train.jsonl`  
  训练集 JSONL 版本（适配 HF / sentence-transformers）
- `eval_report_*.csv` / `pipeline_eval_report.csv`  
  各类评测结果明细
- `pipeline_miss_cases.csv`  
  KG 未解决反例池（L3 主要输入）
- `llm_failures_by_std.csv` / `llm_failures_raw.csv`  
  LLM 失败分析
- `kg_expand_candidates.csv`  
  词表优化候选（下一轮迭代输入）

> 详细字段说明见：`dataset/README.md`

---

## 4. 功能模块说明

### A. 纯 KG 引擎（`app.py`）

提供三层能力：

- `level1_exact`：精确词表匹配
- `level2_fuzzy`：前后缀/字符重叠等模糊匹配
- `level3_graph_reasoning`：图谱规则推理

并提供 API：

- `/api/stats`
- `/api/test_cases`
- `/api/match`
- `/api/batch_eval`
- `/api/generalization`
- `/api/graph_data`

### B. Embedding 引擎（`embedding_service.py`）

- 加载 `Qwen3-Embedding-0.6B`
- 为 `kg_variants.csv` 构建向量索引
- 执行 TopK 相似召回
- 支持不同模型独立索引目录（避免覆盖）

### C. LLM 引擎（`llm_engine.py`）

三种后端可切换：

- `embedding`：最快、成本最低（CPU 可跑）
- `local_qwen`：本地 Qwen3-0.6B 生成模型
- `openai_compat`：OpenAI 兼容接口（DashScope/OpenAI 等）

### D. 混合调度（`pipeline.py`）

标准流程：

1. L1 命中 -> 直接返回
2. L2 高置信命中 -> 直接返回
3. 否则 -> L3 LLM 泛化（基于 KG prompt）
4. 返回结构化结果 + trace

---

## 5. 常用脚本（按使用频率）

### 数据准备

- `scripts/convert_to_csv.py`：JSON -> CSV
- `scripts/generate_training_data.py`：生成 1w embedding 训练样本

### 模型与训练

- `scripts/download_qwen_embedding.py`：下载 embedding 模型
- `scripts/finetune_qwen_embedding.py`：LoRA 微调（sentence-transformers）
- `scripts/compare_ft.py`：base vs ft 对比评测

### 混合链路评测

- `scripts/eval_pipeline.py`：跑 L1/L2/L3 整条链路
- `scripts/analyze_llm_failures.py`：聚合失败，导出词表扩展候选
- `scripts/download_qwen_llm.py`：下载本地 Qwen3-0.6B 生成模型

---

## 6. 快速开始（新同学 10 分钟上手）

### 6.1 安装依赖

```powershell
pip install -r requirements.txt
```

### 6.2 生成/刷新 CSV 数据

```powershell
python scripts/convert_to_csv.py
```

### 6.3 先跑纯 KG 基线

```powershell
python scripts/eval_pipeline.py --no-llm --sample 300
```

### 6.4 跑 KG + LLM（embedding 后端）

```powershell
python scripts/eval_pipeline.py --backend embedding --sample 300
```

### 6.5 导出失败分析，准备词表优化

```powershell
python scripts/analyze_llm_failures.py
```

### 6.6 启动前端可视化页面（对外汇报）

```powershell
python app.py
# 浏览器打开 http://127.0.0.1:8901
```

页面一共 6 个 Tab，汇报时建议按如下顺序演示：

1. **交互式匹配** — 演示一条非标题切换 `纯 KG` / `KG+LLM` 两种引擎，给领导看一条 query 的完整路径（L1→L2→L3）。
2. **KG 基准评估** — 全量 1500 条测试集跑纯 KG，作为基线。
3. **混合链路汇报**（★汇报主页面）— 直接读 `dataset/pipeline_eval_report.csv`，**一屏展示**：
   - 并排大卡片：`纯 KG 49%` → `KG+LLM 91.33%`，`+42.33pp`
   - 来源分布：KG 命中 147 / LLM 救回 127 / LLM 误判 26
   - 按目标层级拆分：L3 纯 KG 0% → 混合 86.7%、out_of_graph 0% → 71.2%
   - 可筛选「LLM 救回」「LLM 误判」「KG 直接命中」查看具体样本
4. **LLM 救回 & 词表优化** — 展示失败反例聚合 + `kg_expand_candidates.csv`，体现"持续迭代闭环"。
5. **80/20 泛化实验** — 证明 KG 词表本身的泛化能力。
6. **图谱浏览器** — 可视化 19 个标准职称 × 629 条映射边。

> **注意**：首次启动时后端会在后台线程加载 Qwen3-Embedding-0.6B，CPU 上约 30-60s，首次索引构建约 2 分钟；头部 badge 会显示 `Embedding 就绪`。在此之前切到"交互式匹配"并选 `纯 KG 规则` 即可立刻使用。
>
> 如果 `models/Qwen3-Embedding-0.6B-ft/` 存在，后端会自动优先加载微调模型，header 会显示 `Embedding 就绪 · 微调模型`；否则回退加载基础模型。

一键端到端健康检查（确认所有接口正常）：

```powershell
python scripts/verify_api.py
```

---

## 7. 评测怎么读（给业务/领导看的指标）

重点看四类数字：

1. **纯 KG 基线准确率**（成本最低能力）
2. **KG 未解决反例数**（LLM 接手规模）
3. **LLM 救回率**（L3 真实价值）
4. **最终 pipeline 准确率提升**（可量化业务收益）

推荐固定汇报口径：

- `KG 命中率`
- `KG 命中准确率`
- `LLM 接手数`
- `LLM 救回率`
- `最终准确率`
- `对比纯 KG 的绝对提升（百分点）`

---

## 8. 词表优化闭环（持续提升效果）

建议每轮迭代按以下步骤：

1. 跑 `eval_pipeline.py` 拿到失败样本
2. 跑 `analyze_llm_failures.py` 汇总盲区
3. 将 `kg_expand_candidates.csv` 喂给 LLM 生成词表补全建议
4. 人工审核后更新 `knowledge_graph_v2.json`
5. 重新评测，比较提升

这就是“**规则引擎 + 大模型引擎共演进**”的核心机制。

---

## 9. 常见问题（FAQ）

### Q1：为什么不直接全量走大模型？

- 成本高、延迟高、可解释性弱。  
建议让 L1/L2 先解决简单且确定的问题，LLM 只处理疑难样本。

### Q2：CPU 能不能跑全流程？

- 推理（embedding backend）可以。
- 0.6B 模型训练在 CPU 上非常慢，建议 GPU 跑微调。

### Q3：怎么快速看项目是否正常？

按顺序跑：

1. `python scripts/convert_to_csv.py`
2. `python scripts/eval_pipeline.py --no-llm --sample 100`
3. `python scripts/eval_pipeline.py --backend embedding --sample 100`

三条都通过，说明数据、规则、embedding 链路基本正常。

---

## 10. 版本与扩展建议

可继续增强的方向：

- 增加 reranker（Qwen3-Reranker）降低错召回
- 增加领域子图（如 AI/金融/制造）做分域路由
- 增加 online hard negative mining 提升 embedding 微调质量
- 增加“拒答/低置信兜底”策略，降低误判风险

---

如果你是第一次接手这个项目，建议先读：

1. 本文档（根 README）
2. `dataset/README.md`（数据字段细节）
3. `pipeline.py` + `scripts/eval_pipeline.py`（主流程）

