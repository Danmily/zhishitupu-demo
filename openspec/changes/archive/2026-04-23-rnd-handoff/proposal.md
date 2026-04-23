# Proposal: 研发交付 — OpenSpec + 业务数据接入

## Why

研发需要可归档、可评审的**规格化交付物**（OpenSpec），并将**真实业务导出**纳入演示工程，便于后续词表扩充、匹配评测与检索扩展。

## What

1. 在仓库根目录建立 `openspec/`，写入当前系统与业务数据资产的 **living specs**。
2. 将人才技能与职称两份 `.xlsx` 导入为 `dataset/business/` 下 CSV + `business_manifest.json`。
3. FastAPI 增加 `/api/business_data/*` 摘要接口；前端增加「业务数据」页签展示高频职称与技能大类样本。

## Scope

- 不包含：自动将全量 10 万+ 职称写入 `knowledge_graph_v2.json`（可作为后续独立 change）。
- 包含：导入脚本、清单 manifest、聚合摘要、API 与 UI 展示。

## Acceptance

- [x] `openspec/specs/` 下至少 2 个能力域 spec 可阅读
- [x] `scripts/ingest_business_excel.py` 可复现生成 `business_manifest.json`
- [x] 启动服务后 `/api/business_data/summary` 返回 `loaded: true`（在已导入前提下）
- [x] 前端「业务数据」页可加载摘要表
