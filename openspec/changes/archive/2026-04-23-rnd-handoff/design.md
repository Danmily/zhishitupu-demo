# Design: 业务数据落盘与 API

## 布局

- `dataset/business/business_manifest.json`：导入元数据、输出相对路径、源文件 `sha256` 前 16 位。
- `dataset/business/talent_skills_full.csv`：`key_name`, `skill_name`（过滤空技能名）。
- `dataset/business/talent_skills_by_key_summary.csv`：每大类行数 + 至多 12 条示例技能（列表序列化在 CSV 单元格内）。
- `dataset/business/talent_titles_full.csv`：`title`。
- `dataset/business/talent_titles_top_freq.csv`：职称 `count` 降序，截断 Top 8000 以降低体积。

## API

- `GET /api/business_data/summary`：直接返回 manifest；不存在时 `loaded: false`。
- `GET /api/business_data/title_freq?limit=`：读取 top_freq CSV。
- `GET /api/business_data/skill_key_summary`：读取 by_key summary（通常仅 5 行，可全量返回）。

## 前端

- 新页签「业务数据」：`fetch` summary + 两张表，避免一次性加载 full CSV。

## 后续扩展（非本 change）

- 从 `talent_titles_top_freq` 抽样生成 `test_cases` 增补用例。
- 与 `skill_graph_v1.json` 按文本做模糊对齐，补全 `related` 边。
