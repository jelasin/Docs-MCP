# docs-mcp

基于 FastMCP + LangChain 的本地文档检索与问答（RAG）工具。

核心能力

- 扫描 `docs/` 下的 md/txt/pdf/docx/csv 文档，分片与向量化；索引持久化到 `db/`（Chroma）
- 通过 MCP 提供工具：`index_docs`（增量索引）、`query_docs`（检索+可选RAG）、`status`、`http_status`、`ingest_urls`
- 嵌入优先使用 OpenAI 兼容接口（`text-embedding-*`），未配置自动回落到 HuggingFace 嵌入模型。
- RAG 基于 LCEL（Prompt + Retriever + ChatOpenAI），可在 `model.conf` 配置模型与 Base URL、API Key。不配置则只返回检索结果。

提示：MCP 是请求-响应模式，无法“把整个向量库传给”客户端；请用 `query_docs` 获取片段或由内置 RAG 生成答案。

## 目录结构

- `docs/` 你的文档目录（md/txt/pdf/docx/csv）
- `db/` Chroma 向量库持久化目录
- `server/docs_server.py` FastMCP 服务器（工具实现均在此）
- `mcp.json` MCP 客户端启动描述
- `model.conf` 模型与服务配置
- `scripts/`（项目自带）增量清单 `docs_done.txt` 会写在这里

## 安装与启动

需要 Python 3.11+

```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# 首次启动需要运行进行初始化，创建 db，docs目录，和初始向量数据库。
# scripts 会生成docs_done.txt文件，增量计算需要向量化的文档，避免重复向量化。
python scripts\index_docs.py --refresh
# 后续仅需运行：
python scripts\index_docs.py
```

启动方式（二选一）：

1. 由 MCP 客户端（例如 VS Code Copilot MCP）通过 `mcp.json` 自动拉起（推荐）

2. 手动运行http服务器（配置http_port和http_host）：

```pwsh
python .\server\docs_server.py
```

HTTP 模式：在 `model.conf` 设置 `http_port`（和可选 `http_host`）后，服务器以 `streamable-http` 方式监听。例如 `http_host=127.0.0.1`，`http_port=3333`。

## 配置（model.conf）

支持 JSON 或 `key=value`。关键项：

- Embeddings （远程 API 调用）：

- - `embed_model`：如 `text-embedding-3-small` 或 HF 模型名（默认回落 `sentence-transformers/all-MiniLM-L6-v2`）
- - `embed_model_baseurl`：OpenAI 兼容基地址（需要 http/https；若缺少 `/v1` 会自动尝试补上）
- - `embed_model_apikey`：API Key（也可用环境变量 `OPENAI_API_KEY`）

- Embeddings（本地调用）：
- - hf_cache_dir：HuggingFace 缓存目录，默认 `models`（会自动创建）
- - hf_model：HuggingFace 模型名（默认 `sentence-transformers/all-MiniLM-L6-v2`）

- RAG（可选）：

- - `rag_model`：如 `gpt-4o`
- - `rag_model_baseurl`：OpenAI 兼容 Chat 接口基地址（必要时自动补 `/v1` 重试）
- - `rag_model_apikey`：API Key（或 `OPENAI_API_KEY`）

- HTTP（可选）：

- - `http_host`：默认 `127.0.0.1`
- - `http_port`：设置后启用 HTTP（`streamable-http`）

示例（ini 风格）：

```ini
# 配置嵌入模型（可选）不配置则下载Hugging Face模型本地调用
# embed_model=text-embedding-3-small
# embed_model_baseurl=
# embed_model_apikey=

# 配置 RAG 模型（可选）不配置则只返回检索结果
# rag_model=gpt-4o
# rag_model_baseurl=
# rag_model_apikey=

# Hugging Face 相关配置
## 模型缓存目录
hf_cache_dir=models
## 备用模型
hf_fallback_model=sentence-transformers/all-MiniLM-L6-v2

# http（可选）
## http_host=0.0.0.0
## http_port=14578
```

## 使用方式（MCP 工具）

- `index_docs(refresh?: boolean)`：构建/更新索引；默认增量。`refresh=true` 会清空 `docs` 集合后重建。
- `query_docs(question: string, top_k?: number=3)`：检索并可选调用 RAG 生成答案。返回字段：
- - `rag_used` 是否调用了外部 RAG
- - `rag_endpoint` 实际调用的基地址（便于诊断 `/v1` 等问题）
- - `answer` 生成答案（若未配置 RAG 则为空）
- - `contexts` 候选片段文本数组
- - `references` 每个片段的路径/URL、chunk 索引与距离
- - `error` 若有错误会给出信息
- `status()`：Chroma 集合信息与路径
- `http_status()`：HTTP 监听信息（启用时）
- `ingest_urls(urls: string[])`：抓取 URL 可见文本并入库

## 说明

- 增量索引：索引记录保存在 `scripts/docs_done.txt`。移动或重命名文件会被视为新文档。
- OpenAI 兼容中转地址：若你的baseurl不带 `/v1`，本项目会在失败时自动追加 `/v1` 重试。
- Windows 上的 HuggingFace 缓存警告：若未启用开发者模式/管理员权限，HF 缓存会提示不支持符号链接，功能不受影响但占用更多空间。若只想屏蔽警告：

```pwsh
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
# 或持久化：
setx HF_HUB_DISABLE_SYMLINKS_WARNING 1
```
