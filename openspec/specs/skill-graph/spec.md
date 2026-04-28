# skill-graph

技能归一化知识图谱：多语言技能根节点树形结构 + 技能搜索 API + 前端技能图谱搜索页签。

## 背景

原始技能词来自业务侧 ASR 导出（`SELECT DISTINCT key_name, skill_name …`），  
词汇量约 1 500+ 条，存在拼写差异、缩写、中英混排等噪音。  
本能力将其归一化为：**根节点（语言/框架）→ 五类工程方向子树** 的层级结构，
嵌入 `skill_graph_v1.json` 作为唯一数据源，并通过 API 和前端统一对外暴露。

---

## Requirements

### Requirement: 多语言技能根节点 skill_tree

系统 SHALL 为以下根节点注入 `skill_tree`（五类方向子树）：  
`skill_python`、`skill_java`、`skill_go`、`skill_javascript`、`skill_typescript`、`skill_rust`、`skill_cpp`

每个根节点的 `skill_tree` SHALL 包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `description` | string | 树的来源与用途说明 |
| `source_excel` | string | 原始 Excel 文件标识 |
| `children` | array | 五个子节点（爬虫/脚本/开发/测试/数据分析） |
| `cross_edges` | array | 跨子树弱边（可选） |

每个子节点 SHALL 包含：`id`、`label`、`edge`（`directed`）、`description`、`aliases`、`excel_top_terms`、`ties`（指向同图其他 skill 节点的弱边）。

#### Scenario: 根节点子树可被 API 获取

- **GIVEN** `skill_graph_v1.json` 中对应根节点已包含 `skill_tree`
- **WHEN** `GET /api/skill_subtrees`
- **THEN** 响应 SHALL 返回所有含 `skill_tree` 的根节点列表，每条包含 `id`、`label`、`category`、`skill_tree`

#### Scenario: Excel 词表归类

- **GIVEN** 业务 Excel 中某行 `key_name` 匹配某根节点（如含 "Python"）
- **WHEN** 执行 `scripts/ingest_excel_skill_trees.py`
- **THEN** 该行 `skill_name` SHALL 按正则桶规则（爬虫/脚本/开发/测试/数据分析）归入对应子节点的 `excel_top_terms`，频次最高的前 `TOP_N`（默认 12）条被保留

---

### Requirement: 技能搜索接口 `/api/skill_search`

系统 SHALL 提供 `POST /api/skill_search`，支持技能关键词的精确 + 模糊 + 图谱扩展 + 可选 Embedding 近邻查询。

**请求体**：

```json
{
  "query": "string",
  "hops": 2,
  "top_related": 10,
  "use_embedding": true
}
```

**响应体**（成功命中）：

```json
{
  "query": "...",
  "matched_node": { "id": "...", "label": "...", "category": "...", "hop": 0 },
  "graph_expansion": [ { "id": "...", "label": "...", "hop": 1 } ],
  "skill_tree": { ... },
  "embedding_job_neighbors": [ { "phrase": "...", "score": 0.9 } ],
  "candidates": []
}
```

**响应体**（模糊候选）：

```json
{
  "query": "...",
  "matched_node": null,
  "candidates": [ { "id": "...", "label": "...", "score": 0.85 } ]
}
```

#### Scenario: 精确命中

- **GIVEN** 查询词与某节点 `label` 或 `aliases` 完全匹配（大小写不敏感）
- **WHEN** `POST /api/skill_search` with `{"query": "Python"}`
- **THEN** `matched_node.hop == 0`，`graph_expansion` 包含 2 跳内相邻节点，`skill_tree` 为该根节点完整子树

#### Scenario: 模糊匹配

- **GIVEN** 查询词与任何节点不精确匹配，但与某节点 token 重叠
- **WHEN** `POST /api/skill_search` with `{"query": "Pyhton"}`（拼写错误）
- **THEN** `matched_node == null`，`candidates` 列表按 token 相似度降序，至多返回 10 条

#### Scenario: Embedding 近邻（可选）

- **GIVEN** `use_embedding == true` 且 Embedding 引擎已加载
- **WHEN** 精确命中某节点
- **THEN** 响应额外包含 `embedding_job_neighbors`（最多 5 条相关岗位标准表述）

---

### Requirement: 前端技能图谱搜索页签

系统 SHALL 在前端导航中提供「技能图谱」页签（`data-t="skills"`），替代已删除的「技能扩展 & ES DSL」和「业务数据」页签。

页签 SHALL 包含：

1. **搜索框** + 快捷示例词（Python、Java、Go 等）
2. **匹配节点卡片**（`#skMatchCard`）：展示命中节点信息
3. **图谱扩展卡片**（`#skGraphCard`）：展示多跳邻居，按 hop 着色区分
4. **技能树卡片**（`#skTreeCard`）：展示根节点五类子树
5. **Embedding 近邻卡片**（`#skEmbCard`）：展示相关岗位
6. **候选词卡片**（`#skCandCard`）：无精确匹配时展示模糊候选

「图谱浏览器」页签 SHALL 额外在 `#skillSubtreesCard` 区域展示所有多语言子树的可折叠详情（`<details>`/`<summary>` 组件），数据来自 `GET /api/skill_subtrees`。

#### Scenario: 首次打开技能图谱页签

- **GIVEN** 用户点击「技能图谱」导航项
- **WHEN** 页签切换
- **THEN** 显示空状态提示「输入技能关键词开始探索…」，快捷示例词可点击直接触发搜索

---

## 相关实现

| 文件 | 说明 |
|------|------|
| `skill_graph_v1.json` | 所有根节点已注入 `skill_tree`（7 种语言/框架） |
| `app.py` | 新增 `POST /api/skill_search`、`GET /api/skill_subtrees` |
| `query_expansion.py` | `SkillGraph` 类（图谱遍历 + 模糊匹配）、`_do_expand_query` |
| `scripts/ingest_excel_skill_trees.py` | 通用多语言 Excel 词表 → skill_tree 脚本 |
| `dataset/skill_trees_from_excel.json` | 多语言 skill_tree 快照（参考/调试用） |
| `index.html` | 新增「技能图谱」页签及相关 UI 组件 |
