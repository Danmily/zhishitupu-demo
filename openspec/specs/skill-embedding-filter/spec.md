# skill-embedding-filter

语义岗位扩展（`semantic_expansions`）的 **标准 ID 过滤**增强：  
限制 Embedding 近邻结果只在与主匹配岗位相同的标准职类内搜索，消除跨类语义噪音。

## 背景

`/api/pipeline_match` 在 L3 语义扩展阶段对整个 `kg_variants.csv` 做向量近邻，  
导致"前端人员"等查询返回"C端产品经理"等表面相似但语义无关的结果。  
根源：Qwen3-Embedding 模型将"前端"与"C端"的"端"字赋予了相近的上下文向量。

本能力在 `HybridPipeline._semantic_title_expansions` 中增加 **`matched_standard_id` 过滤参数**，  
使扩展结果严格限定在主匹配岗位的标准职类变体集合内。

---

## Requirements

### Requirement: 语义扩展结果限定标准 ID

当主匹配链路（L1/L2/L3）成功确定 `standard_id` 时，系统 SHALL 在后续 Embedding 近邻搜索中过滤掉所有 `standard_id` 不符的候选，保证扩展词在语义上属于同一岗位类。

#### Scenario: 过滤生效——"前端人员"不再返回产品经理

- **GIVEN** 主匹配结果 `standard_id == "frontend_engineer"`
- **WHEN** 调用 `_semantic_title_expansions(title="前端人员", matched_standard_id="frontend_engineer")`
- **THEN** 返回列表中不包含任何 `standard_id == "product_manager"` 的候选，  
  结果条目均为前端工程师相关的表述（如"前端开发工程师"、"Web 前端"等）

#### Scenario: 未命中时不过滤

- **GIVEN** 主匹配结果为 `L0_miss`（`standard_id` 为空）
- **WHEN** 调用 `_semantic_title_expansions(title="…", matched_standard_id=None)`
- **THEN** 过滤逻辑不启用，返回全局 top-K 近邻（保持原有行为）

#### Scenario: 扩大初始搜索池以补偿过滤损耗

- **GIVEN** `matched_standard_id` 非空
- **WHEN** 调用 `_semantic_title_expansions`
- **THEN** 内部向量搜索的 `top_k` SHALL 为 `max(max_n * 6, 32)`（默认约为 30），  
  远大于无过滤时的 `max(max_n * 3, 16)`，确保过滤后仍有足够候选填满 `max_n`

### Requirement: 语义扩展条目数上限

`/api/pipeline_match` 响应中 `semantic_expansions` SHALL 至多返回 **5 条**（`max_n=5`）。

#### Scenario: 截断超量结果

- **GIVEN** 过滤后命中 20 条候选
- **WHEN** `max_n == 5`
- **THEN** 响应中 `semantic_expansions` 长度 ≤ 5，按相似度降序保留前 5 条

---

## 实现细节

### `pipeline.py — _semantic_title_expansions`

```python
def _semantic_title_expansions(
    self,
    title: str,
    context: str = "",
    max_n: int = 5,
    matched_standard_id: Optional[str] = None,
) -> list[dict]:
    pool_size = max(max_n * 6, 32) if matched_standard_id else max(max_n * 3, 16)
    hits = self.llm.ekg.search(q, top_k=pool_size)
    for h in hits:
        if matched_standard_id and h.standard_id != matched_standard_id:
            continue
        # … 去重、得分截断、append …
    return out[:max_n]
```

### `pipeline.py — _attach_expansions`

```python
def _attach_expansions(self, result: dict, title: str, context: str) -> dict:
    matched_sid: Optional[str] = result.get("standard_id") or None
    ex = self._semantic_title_expansions(title, context, matched_standard_id=matched_sid)
    result["semantic_expansions"] = ex
    return result
```

---

## 相关实现

| 文件 | 位置 | 说明 |
|------|------|------|
| `pipeline.py` | `_semantic_title_expansions` | 新增 `matched_standard_id` 参数及过滤逻辑 |
| `pipeline.py` | `_attach_expansions` | 向 `_semantic_title_expansions` 传递主匹配的 `standard_id` |
