# kg-job-matching

猎头 / 人才场景下的**职称标准化**能力：多层规则 + 可选 Embedding 泛化。

## Requirements

### Requirement: 多层匹配降级路径

系统 SHALL 按顺序尝试 L1 精确匹配、L2 模糊匹配；若仍未满足置信度或未命中，且已配置 Embedding 引擎，则 SHALL 走 L3 向量近邻分类。

#### Scenario: L1 命中变体

- **GIVEN** 输入职称与知识图谱中某变体经规范化后一致
- **WHEN** 调用匹配接口
- **THEN** 返回标准职称、层级 `L1_exact`、置信度 1.0 或剥离修饰后的略低置信度

#### Scenario: L3 语义扩展展示

- **GIVEN** 混合链路（HybridPipeline）已加载 Embedding 索引
- **WHEN** 调用 `/api/pipeline_match`
- **THEN** 响应 SHALL 包含 `semantic_expansions` 列表（至多约 20 条近邻岗位表述），用于检索扩展与产品展示

### Requirement: 纯 KG 与混合链路并存

系统 SHALL 提供纯知识图谱匹配接口（无模型依赖）与混合链路接口（依赖本地 Qwen3-Embedding），并在前端允许切换引擎模式。

## 相关实现

- `app.py`：`TitleKnowledgeGraph`、`/api/match`、`/api/batch_eval`
- `pipeline.py`：`HybridPipeline`、`/api/pipeline_match`
- `knowledge_graph_v2.json`：标准岗与变体
