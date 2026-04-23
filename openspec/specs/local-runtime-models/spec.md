# local-runtime-models

说明：**向量模型与生成模型权重不入 Git**，由 `.gitignore` 排除；克隆仓库后在本机或内网镜像拉取。

## Requirements

### Requirement: 权重不得进入版本库

仓库 SHALL 在 `.gitignore` 中忽略以下路径（或等价目录名），避免误提交大文件：

- `models/Qwen3-0.6B/` — 生成模型（`LocalQwenEngine` 可选）
- `models/Qwen3-Embedding-0.6B/` — 向量模型（混合链路 L3 / `embedding_service` 默认）
- `models/Qwen3-Embedding-0.6B-ft/` — 微调后的向量模型（若存在则 `app.py` 优先加载）
- `models/_kg_index/` — 变体向量索引缓存（可重建）
- `checkpoints/` — 训练检查点

#### Scenario: 协作者克隆后具备 Embedding 能力

- **GIVEN** 已安装 `pip install -r requirements.txt`
- **WHEN** 执行 `python scripts/download_qwen_embedding.py`（国内可设 `HF_ENDPOINT=https://hf-mirror.com`）
- **THEN** 本地出现 `models/Qwen3-Embedding-0.6B/`，启动 `python app.py` 后混合链路可加载（首次可能构建 `models/_kg_index/`）

#### Scenario: 可选安装生成模型

- **WHEN** 需要 `llm_engine.LocalQwenEngine`（非默认路径）
- **THEN** 执行 `python scripts/download_qwen_llm.py`，得到 `models/Qwen3-0.6B/`

## 命令速查

```powershell
# Embedding（推荐人人执行）
$env:HF_ENDPOINT='https://hf-mirror.com'   # 可选
python scripts/download_qwen_embedding.py

# 生成模型 Qwen3-0.6B（仅在用 local_qwen 时需要）
python scripts/download_qwen_llm.py
```

首次向量检索前，应用侧会通过 `embedding_service.EmbeddingKG.build_index()` 生成索引；若已存在 `models/_kg_index/<tag>/` 则直接加载。

## 与 README 的关系

根目录 `README.md` 中「研发交付 / Git 与模型」小节为对外简述；**本 spec 为交付与评审用的完整约定**。
