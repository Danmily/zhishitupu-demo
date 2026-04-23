# business-talent-exports

研发提供的**真实业务导出**（人才画像技能维度 + 职称明细）在项目内的存储、校验与对外摘要 API。

## Requirements

### Requirement: 可复现导入

系统 SHALL 提供脚本 `scripts/ingest_business_excel.py`，从 `.xlsx` 生成 UTF-8 CSV 与 `business_manifest.json`（行数、去重口径、源文件指纹）。

#### Scenario: 导入技能宽表

- **GIVEN** 技能导出包含列 `key_name`, `skill_name`
- **WHEN** 执行导入脚本并指定 `--skills`
- **THEN** 生成 `dataset/business/talent_skills_full.csv` 与按 `key_name` 聚合的 `talent_skills_by_key_summary.csv`

#### Scenario: 导入职称列

- **GIVEN** 职称导出包含列 `title`
- **WHEN** 执行导入脚本并指定 `--titles`
- **THEN** 生成 `dataset/business/talent_titles_full.csv` 与 `talent_titles_top_freq.csv`（高频职称 Top N）

### Requirement: 运行时摘要

当 `dataset/business/business_manifest.json` 存在时，HTTP API SHALL 暴露业务资产摘要与高频样本，供演示与研发验收；若未导入则 SHALL 返回明确提示而非 500。

## 数据说明（当前样本）

- 技能：`key_name` 为能力大类（如 `skill.engineering_ability`、`skill.ai_ability`、`skill.tool` 等），`skill_name` 为具体技能/工具表述。
- 职称：与 talent profile 关联的去重/清洗后的 `title` 文本，可用于词表扩充与匹配压测抽样。

## 隐私与合规

业务 CSV 可能含敏感字段聚合；交付物 SHALL 仅包含研发明确提供的导出；仓库提交前由数据方确认是否保留 `*_full.csv` 或仅保留聚合文件。
