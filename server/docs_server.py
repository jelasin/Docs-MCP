import os
import sys
import json
import glob
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Annotated

import numpy as np
from typing import cast

# Vector DB (Chroma)
try:
	from chromadb import PersistentClient
except Exception:
	chromadb = None
	PersistentClient = None  # type: ignore

try:
	from sentence_transformers import SentenceTransformer
except Exception:
	SentenceTransformer = None  # Will validate at startup

# LangChain related (optional; we guard usage at runtime)
try:
	from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # type: ignore
	from langchain_chroma import Chroma as LCChroma  # type: ignore
	from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
	from langchain_core.prompts import ChatPromptTemplate  # type: ignore
	from langchain_core.output_parsers import StrOutputParser  # type: ignore
	from langchain_core.runnables import RunnablePassthrough  # type: ignore
	from langchain.chat_models import init_chat_model  # type: ignore
except Exception:
	ChatOpenAI = None  # type: ignore
	OpenAIEmbeddings = None  # type: ignore
	LCChroma = None  # type: ignore
	HuggingFaceEmbeddings = None  # type: ignore
	ChatPromptTemplate = None  # type: ignore
	StrOutputParser = None  # type: ignore
	RunnablePassthrough = None  # type: ignore
	init_chat_model = None  # type: ignore

try:
	from fastmcp import FastMCP
except Exception:
	FastMCP = None  # type: ignore[assignment]

from pydantic import BaseModel, Field, SecretStr


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
DB_DIR = os.path.join(BASE_DIR, "db")
CONF_PATH = os.path.join(BASE_DIR, "model.conf")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
PROCESSED_PATH = os.path.join(SCRIPTS_DIR, "docs_done.txt")

def _normalize_base_url(url: Optional[str]) -> Optional[str]:
	"""Ensure base_url has http/https scheme; return None if empty."""
	if not url:
		return None
	u = str(url).strip()
	if not u:
		return None
	if not (u.startswith("http://") or u.startswith("https://")):
		u = "http://" + u
	return u


def ensure_dirs():
	# Always ensure docs/ and db/ exist
	os.makedirs(DOCS_DIR, exist_ok=True)
	os.makedirs(DB_DIR, exist_ok=True)
	# scripts/ 目录为项目自带，不主动创建
	# 主动创建 HuggingFace 缓存目录（例如 model.conf: hf_cache_dir=models）
	try:
		conf = load_conf()
		cache_dir = conf.get("hf_cache_dir") or None
		if cache_dir:
			cache_path = cache_dir if os.path.isabs(cache_dir) else os.path.join(BASE_DIR, cache_dir)
			os.makedirs(cache_path, exist_ok=True)
	except Exception:
		# 配置缺失或权限问题不应阻断主流程
		pass


def _load_done() -> Dict[str, Dict[str, int]]:
	"""Load processed file records from scripts/docs_done.txt.

	Format: TSV lines -> path\tmtime\tsize
	Return: { path: {"mtime": int, "size": int} }
	"""
	rec: Dict[str, Dict[str, int]] = {}
	try:
		if os.path.exists(PROCESSED_PATH):
			with open(PROCESSED_PATH, "r", encoding="utf-8") as f:
				for line in f:
					line = line.strip()
					if not line or line.startswith("#"):
						continue
					parts = line.split("\t")
					if len(parts) >= 3:
						p, m, s = parts[0], parts[1], parts[2]
						try:
							rec[p] = {"mtime": int(m), "size": int(s)}
						except Exception:
							continue
	except Exception:
		pass
	return rec


def _save_done(rec: Dict[str, Dict[str, int]]) -> None:
	"""Persist processed file records to scripts/docs_done.txt (TSV)."""
	lines = [f"{p}\t{meta.get('mtime', 0)}\t{meta.get('size', 0)}" for p, meta in sorted(rec.items())]
	try:
		with open(PROCESSED_PATH, "w", encoding="utf-8") as f:
			f.write("\n".join(lines) + ("\n" if lines else ""))
	except Exception:
		pass


def load_conf() -> Dict[str, Any]:
	# model.conf can be JSON or simple key=value lines
	if not os.path.exists(CONF_PATH):
		return {}
	try:
		with open(CONF_PATH, "r", encoding="utf-8") as f:
			txt = f.read().strip()
			if not txt:
				return {}
			if txt.lstrip().startswith("{"):
				return json.loads(txt)
			conf: Dict[str, Any] = {}
			for line in txt.splitlines():
				line = line.strip()
				if not line or line.startswith("#"):
					continue
				if "=" in line:
					k, v = line.split("=", 1)
					conf[k.strip()] = v.strip()
			return conf
	except Exception:
		return {}


@dataclass
class DocChunk:
	id: str
	doc_path: str
	chunk_index: int
	text: str


def _get_embed_model_name(conf: Dict[str, Any]) -> str:
	return conf.get("embed_model") or conf.get("embedding_model") or "sentence-transformers/all-MiniLM-L6-v2"


def _build_langchain_components(conf: Dict[str, Any]):
	"""Create LangChain embeddings, vectorstore retriever, and LLM based on model.conf.

	Supports:
	- OpenAI embeddings if embed_model like "text-embedding-*" and API key present
	- SentenceTransformer embeddings otherwise (for ingestion only; retrieval via Chroma uses stored vectors)
	- ChatOpenAI LLM per rag_model + baseurl/apikey
	"""
	if LCChroma is None:
		raise RuntimeError("LangChain 未安装，请在 requirements.txt 中包含 langchain、langchain-openai、langchain-community、openai 并安装依赖。")

	# Use the same embeddings logic as ingestion (with fallback)
	lc_embeddings = _make_lc_embeddings(conf)

	# Vector store: point to the same Chroma persisted directory and collection name
	retriever = LCChroma(
		collection_name="docs",
		persist_directory=DB_DIR,
		embedding_function=lc_embeddings,
	).as_retriever(search_kwargs={"k": 3})

	# LLM（带 /v1 回退与健康探测），使用统一的 init_chat_model 接口
	rag_model = conf.get("rag_model") or "gpt-4o"
	rag_base_raw = _normalize_base_url(conf.get("rag_model_baseurl") or conf.get("rag_baseurl"))
	rag_key = conf.get("rag_model_apikey") or os.getenv("OPENAI_API_KEY")
	if init_chat_model is None:
		raise RuntimeError("LangChain init_chat_model 未安装或不可用（请升级 langchain 至支持该接口的版本）")
	if not rag_key:
		# 允许无 key 情况下仅返回检索，不做生成
		llm = None
		return retriever, llm

	def _make_llm(base_url: Optional[str]):
		# 统一入口，必要时可通过 rag_model_provider 指定提供方（如 openai、anthropic、groq 等）
		provider = conf.get("rag_model_provider") or None
		kwargs: Dict[str, Any] = {"model": rag_model, "temperature": 0}
		if provider:
			kwargs["model_provider"] = provider
		if base_url:
			kwargs["base_url"] = base_url
		if rag_key:
			kwargs["api_key"] = str(rag_key)
		return cast(Any, init_chat_model)(**kwargs)

	def _probe_llm(test_llm):
		try:
			_ = test_llm.invoke("healthcheck")
			return True
		except Exception:
			return False

	# 尝试 1：按原始 base_url 构建并探测
	llm: Optional[Any] = None
	try_base = rag_base_raw
	try:
		cand = _make_llm(try_base)
		if _probe_llm(cand):
			llm = cand
			setattr(llm, "_docs_mcp_rag_endpoint", try_base)
			setattr(llm, "_docs_mcp_model_name", f"{rag_model} (init_chat_model)")
			return retriever, llm
	except Exception:
		pass

	# 尝试 2：如果 base_url 缺少 /v1，则自动补上并再探测
	if try_base and not str(try_base).rstrip("/").endswith("v1"):
		try_v1 = str(try_base).rstrip("/") + "/v1"
		try:
			cand2 = _make_llm(try_v1)
			if _probe_llm(cand2):
				llm = cand2
				setattr(llm, "_docs_mcp_rag_endpoint", try_v1)
				setattr(llm, "_docs_mcp_model_name", f"{rag_model} (init_chat_model)")
				return retriever, llm
		except Exception:
			pass

	# 如果仍失败，抛出异常，让调用方回退为仅检索（并显示错误）
	raise RuntimeError("RAG LLM 初始化失败：请检查 rag_model_baseurl/baseurl 与 API Key，必要时在地址末尾添加 /v1")


def _make_lc_embeddings(conf: Dict[str, Any]):
	"""Return a LangChain embeddings object per model.conf.

	Priority:
	- OpenAIEmbeddings if embed_model like text-embedding-* and API key present
	- HuggingFaceEmbeddings (SentenceTransformers) otherwise
	"""
	model_name = _get_embed_model_name(conf)
	embed_base = _normalize_base_url(conf.get("embed_model_baseurl") or conf.get("embedding_baseurl"))
	embed_key = conf.get("embed_model_apikey") or conf.get("embedding_apikey") or os.getenv("OPENAI_API_KEY")
	attempted_openai = False
	# Try OpenAI-compatible embeddings if configured and key present
	if (
		OpenAIEmbeddings is not None
		and (model_name.startswith("text-embedding") or model_name in {"text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"})
		and embed_key
	):
		attempted_openai = True
		# Attempt 1: use provided base_url as-is
		try:
			oe = OpenAIEmbeddings(model=model_name, base_url=embed_base, api_key=SecretStr(str(embed_key)))
			_ = oe.embed_documents(["healthcheck"])  # probe
			setattr(oe, "_docs_mcp_model_name", f"{model_name} (OpenAI)")
			return oe
		except Exception:
			# Attempt 2: append '/v1' to base_url when missing (some providers require this)
			try:
				if embed_base and not str(embed_base).rstrip("/").endswith("v1"):
					oe = OpenAIEmbeddings(model=model_name, base_url=str(embed_base).rstrip("/") + "/v1", api_key=SecretStr(str(embed_key)))
					_ = oe.embed_documents(["healthcheck"])  # probe
					setattr(oe, "_docs_mcp_model_name", f"{model_name} (OpenAI)")
					return oe
			except Exception:
				pass
	# Fallback to local sentence-transformers via HuggingFaceEmbeddings
	# If an OpenAI-like name is set or remote attempt failed, use a solid local default
	if "text-embedding" in model_name or attempted_openai:
		model_name = conf.get("hf_fallback_model") or "sentence-transformers/all-MiniLM-L6-v2"
	if HuggingFaceEmbeddings is None:
		raise RuntimeError("HuggingFaceEmbeddings 不可用，请安装 langchain-community 和 sentence-transformers。")
	cache_dir = conf.get("hf_cache_dir") or None
	try:
		hf = HuggingFaceEmbeddings(model_name=model_name, cache_folder=cache_dir)
	except TypeError:
		# Older/newer versions may not support cache_folder param
		hf = HuggingFaceEmbeddings(model_name=model_name)
	setattr(hf, "_docs_mcp_model_name", f"{model_name} (HF)")
	return hf


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
	words = text.split()
	chunks = []
	i = 0
	while i < len(words):
		chunk = words[i:i + chunk_size]
		if not chunk:
			break
		chunks.append(" ".join(chunk))
		i += chunk_size - overlap
		if i <= 0:
			i += chunk_size
	return chunks


def read_doc(path: str) -> str:
	ext = os.path.splitext(path)[1].lower()
	if ext in (".md", ".txt"):
		with open(path, "r", encoding="utf-8", errors="ignore") as f:
			return f.read()
	if ext == ".csv":
		# 解析 CSV 为可检索文本：优先映射成 "col=value" 形式的行，便于语义检索
		import csv
		try:
			with open(path, "r", encoding="utf-8", errors="ignore") as f:
				sample = f.read(8192)
				f.seek(0)
				try:
					dialect = csv.Sniffer().sniff(sample)
				except Exception:
					dialect = csv.excel
				try:
					has_header = csv.Sniffer().has_header(sample)
				except Exception:
					has_header = True
				reader = csv.reader(f, dialect)
				headers = None
				lines: List[str] = []
				for i, row in enumerate(reader):
					if i == 0 and has_header:
						headers = row
						continue
					if headers and len(headers) == len(row):
						pairs = [f"{h}={v}" for h, v in zip(headers, row)]
						lines.append(" | ".join(pairs))
					else:
						lines.append(",".join(row))
				return "\n".join(lines)
		except Exception as e:
			return f"[CSV读取失败] {e}"
	if ext == ".pdf":
		try:
			from pypdf import PdfReader
			reader = PdfReader(path)
			return "\n".join(page.extract_text() or "" for page in reader.pages)
		except Exception as e:
			return f"[PDF读取失败] {e}"
	if ext in (".docx",):
		try:
			from docx import Document  # python-docx
			d = Document(path)
			return "\n".join(p.text for p in d.paragraphs)
		except Exception as e:
			return f"[DOCX读取失败] {e}"
	# Unsupported -> raw bytes decode
	try:
		with open(path, "rb") as f:
			return f.read().decode("utf-8", errors="ignore")
	except Exception:
		return ""


class Embedder:
	def __init__(self, model_name: Optional[str] = None):
		if SentenceTransformer is None:
			raise RuntimeError("sentence-transformers 未安装。请先安装依赖。")
		self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
		self.model = SentenceTransformer(self.model_name)

	def encode(self, texts: List[str]) -> np.ndarray:
		embs = self.model.encode(texts, normalize_embeddings=False, convert_to_numpy=True)
		return embs.astype(np.float32)


if FastMCP is not None:
	app = FastMCP(name="docs-mcp")
else:
	# Provide a no-op shim for decorators so the module can be imported without fastmcp
	def _noop_decorator(*dargs, **dkwargs):
		"""A flexible no-op decorator supporting @app.tool and @app.tool(...)."""
		# If used as bare decorator: @app.tool
		if dargs and callable(dargs[0]) and len(dargs) == 1 and not dkwargs:
			return dargs[0]
		# If used as parameterized decorator: @app.tool(...)
		def deco(fn):
			return fn
		return deco

	class _AppShim:
		tool = staticmethod(_noop_decorator)

		def run(self, *args, **kwargs):  # no-op for type checking
			print("FastMCP 未安装，无法运行服务器。")

	app = _AppShim()  # type: ignore[assignment]


def run_index_docs(refresh: bool = False) -> str:
	"""
	扫描 docs/ 下的 pdf/docx/md/txt 文档，切分并向量化，存入 db/。
	参数: refresh=true 将清空并重建向量库。
	返回: 处理的文件及分片统计。
	"""
	ensure_dirs()
	if refresh:
		# 清理旧的 Chroma 集合与已处理清单
		try:
			if os.path.exists(PROCESSED_PATH):
				os.remove(PROCESSED_PATH)
		except Exception:
			pass

	conf = load_conf()
	# handle refresh by deleting collection first
	if refresh and PersistentClient is not None:
		try:
			client = PersistentClient(path=DB_DIR)
			client.delete_collection("docs")
		except Exception:
			pass

	# Build LangChain embeddings and vectorstore
	if LCChroma is None:
		raise RuntimeError("LangChain 未安装，请在 requirements.txt 中包含 langchain、langchain-openai、langchain-community、openai 并安装依赖。")
	lc_embeddings = _make_lc_embeddings(conf)
	vectorstore = LCChroma(collection_name="docs", persist_directory=DB_DIR, embedding_function=lc_embeddings)

	exts = ["**/*.md", "**/*.txt", "**/*.pdf", "**/*.docx", "**/*.csv"]
	files: List[str] = []
	for pattern in exts:
		files.extend(glob.glob(os.path.join(DOCS_DIR, pattern), recursive=True))

	# 增量：仅处理新增或修改过的文件
	done = _load_done()
	to_process: List[str] = []
	skipped = 0
	for fp in sorted(set(files)):
		rel = os.path.relpath(fp, BASE_DIR)
		try:
			st = os.stat(fp)
			cur = {"mtime": int(st.st_mtime), "size": int(st.st_size)}
			prev = done.get(rel)
			if refresh or prev is None or prev.get("mtime") != cur["mtime"] or prev.get("size") != cur["size"]:
				to_process.append(fp)
				# 对已存在的旧向量，先删除再写入，避免重复
				try:
					vectorstore.delete(where={"doc_path": rel})
				except Exception:
					pass
				# 更新记录
				done[rel] = cur
			else:
				skipped += 1
		except Exception:
			to_process.append(fp)

	total_chunks = 0
	t0 = time.time()
	for fp in to_process:
		text = read_doc(fp)
		chunks = chunk_text(text)
		if not chunks:
			continue
		rel = os.path.relpath(fp, BASE_DIR)
		ids = [f"{rel}::chunk-{i}" for i in range(len(chunks))]
		metadatas: List[Dict[str, Any]] = [{"doc_path": rel, "chunk_index": i} for i in range(len(chunks))]
		# Use LangChain vectorstore to upsert texts
		vectorstore.add_texts(texts=chunks, metadatas=cast(Any, metadatas), ids=ids)
		total_chunks += len(chunks)

	# 保存已处理清单
	_save_done(done)

	dt = time.time() - t0
	used_model = getattr(lc_embeddings, "_docs_mcp_model_name", _get_embed_model_name(conf))
	# Chroma auto-persists in modern versions; no manual persist required
	return (
		f"已索引文件数: {len(set(files))}, 新处理文件: {len(to_process)}, 跳过未改动: {skipped}, 生成分片: {total_chunks}, 用时: {dt:.2f}s"
		f"（模型: {used_model}，清单: {os.path.relpath(PROCESSED_PATH, BASE_DIR)}，LangChain->Chroma）"
	)


class Reference(BaseModel):
	path: Optional[str] = Field(default=None, description="Relative path of the local document")
	url: Optional[str] = Field(default=None, description="Source URL")
	chunk_index: int = Field(default=0, ge=0, description="Chunk index")
	dist: Optional[float] = Field(default=None, description="Retrieval distance (lower is more relevant)")
	text: str = Field(description="Matched text chunk")


class RagAnswer(BaseModel):
	question: str = Field(description="User question")
	top_k: int = Field(description="Number of returned chunks")
	rag_used: bool = Field(description="Whether an external RAG endpoint was invoked")
	rag_endpoint: Optional[str] = Field(default=None, description="RAG endpoint URL")
	answer: Optional[str] = Field(default=None, description="Answer generated by the RAG model")
	contexts: List[str] = Field(default_factory=list, description="Candidate context chunks (plain text)")
	references: List[Reference] = Field(default_factory=list, description="Source metadata for each chunk")
	error: Optional[str] = Field(default=None, description="Error message, if any (e.g., RAG failure)")


@app.tool(
	name="index_docs",
	description="Scan docs/ and (re)build the Chroma vector DB using LangChain. Supports incremental by default.",
	tags={"index", "chroma", "docs"},
)  # type: ignore[union-attr]
def index_docs(
	refresh: Annotated[bool, Field(description="If true, drop and rebuild the collection")] = False,
) -> str:
	"""Expose run_index_docs as an MCP tool so clients can trigger indexing without scripts."""
	try:
		return run_index_docs(refresh=bool(refresh))
	except Exception as e:
		return f"索引失败: {e}"


@app.tool(
	name="query_docs",
	description="Search the local Chroma vector DB and optionally call a configured RAG endpoint to produce an answer.",
	tags={"docs", "search", "rag"},
)  # type: ignore[union-attr]
def query_docs(
	question: Annotated[str, Field(description="User question in natural language")],
	top_k: Annotated[int, Field(ge=1, le=50, description="Number of chunks to return")]=3,
) -> RagAnswer:
	"""
	Retrieve top-k relevant chunks from the vector DB and optionally call a configured RAG endpoint
	(model.conf: {"rag_endpoint": "..."}) to generate a final answer. MCP returns only results/answers,
	not the entire vector database.
	"""
	ensure_dirs()
	conf = load_conf()

	# Use LangChain vectorstore for retrieval so embeddings fallback works (OpenAI or HF)
	if LCChroma is None:
		return RagAnswer(question=question, top_k=top_k, rag_used=False, rag_endpoint=None, contexts=[], references=[], answer=None, error="LangChain 未安装或不可用。")
	lc_embeddings = _make_lc_embeddings(conf)
	vectorstore = LCChroma(collection_name="docs", persist_directory=DB_DIR, embedding_function=lc_embeddings)
	try:
		pairs = cast(List[Any], vectorstore.similarity_search_with_score(question, k=top_k))
	except Exception as e:
		return RagAnswer(question=question, top_k=top_k, rag_used=False, rag_endpoint=None, contexts=[], references=[], answer=None, error=f"检索失败: {e}")
	if not pairs:
		return RagAnswer(question=question, top_k=top_k, rag_used=False, rag_endpoint=None, contexts=[], references=[], answer=None, error="没有检索到相关内容。请先执行 index_docs。")

	docs: List[str] = []
	references: List[Reference] = []
	for doc, score in pairs:
		text = getattr(doc, "page_content", "")
		meta = getattr(doc, "metadata", {}) or {}
		m_path = meta.get("doc_path") if isinstance(meta, dict) else None
		m_url = meta.get("source_url") if isinstance(meta, dict) else None
		m_idx = meta.get("chunk_index") if isinstance(meta, dict) else 0
		try:
			idx_val = int(m_idx) if m_idx is not None else 0
		except Exception:
			idx_val = 0
		dist_val: Optional[float] = float(score) if score is not None else None
		docs.append(text)
		references.append(Reference(path=str(m_path) if isinstance(m_path, str) else None,
			url=str(m_url) if isinstance(m_url, str) else None,
			chunk_index=idx_val, dist=dist_val, text=text))

	# Option 1: if configured with LLM credentials, run an LCEL Runnable chain to generate an answer
	rag_used = False
	rrag_endpoint = None
	answer_text: Optional[str] = None
	err_msg: Optional[str] = None
	try:
		retriever, llm = _build_langchain_components(conf)
		if llm is not None and ChatPromptTemplate is not None and StrOutputParser is not None and RunnablePassthrough is not None:
			prompt = ChatPromptTemplate.from_messages([
				("system", "You are a helpful assistant. Use the provided context to answer the question concisely. If unsure, say you don't know."),
				("human", "Question: {question}\n\nContext:\n{context}")
			])
			def _format_docs(docs: List[Any]) -> str:
				try:
					return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)
				except Exception:
					return ""
			# Build chain and attempt LLM call
			chain = (
				{"context": retriever | _format_docs, "question": RunnablePassthrough()}
				| prompt
				| llm
				| StrOutputParser()
			)
			try:
				answer_text = chain.invoke(question)
				rag_used = True
				# 如果 _build_langchain_components 设置了端点，读取它；否则回退配置值
				endpoint = getattr(llm, "_docs_mcp_rag_endpoint", None)
				rrag_endpoint = endpoint or _normalize_base_url(conf.get("rag_model_baseurl") or conf.get("rag_baseurl"))
			except Exception as e1:
				# 将错误记录下来，但不影响检索结果返回
				err_msg = str(e1)
		elif llm is None:
			# LLM 未配置，保持仅检索
			err_msg = "未配置 rag_model_apikey，已返回检索上下文"
	except Exception as e:
		# 不影响检索结果返回
		err_msg = str(e)

	return RagAnswer(
		question=question,
		top_k=top_k,
		rag_used=rag_used,
		rag_endpoint=rrag_endpoint,
		answer=answer_text,
		contexts=docs,
		references=references,
		error=err_msg,
	)


@app.tool(
	name="status",
	description="Basic health check and Chroma collection info for this server.",
	tags={"health", "chroma"},
)  # type: ignore[union-attr]
def status() -> str:
	"""Return basic availability info and the Chroma collection path."""
	ensure_dirs()
	if PersistentClient is None:
		return "Chroma 未安装"
	try:
		client = PersistentClient(path=DB_DIR)
		cols = [c.name for c in client.list_collections()]
		exists = "docs" in cols
		return f"Chroma 集合: docs ({'存在' if exists else '未创建'}); 路径: {DB_DIR}"
	except Exception:
		return f"Chroma 集合: docs; 路径: {DB_DIR}"

@app.tool(
	name="http_status",
	description="Show HTTP server status when enabled via model.conf (http_port). Uses streamable-http.",
	tags={"health", "http"},
)  # type: ignore[union-attr]
def http_status() -> str:
	"""Return HTTP server info (host/port) if enabled via model.conf (http_port)."""
	conf = load_conf()
	port = int(conf.get("http_port", 0) or 0)
	if port > 0:
		host = str(conf.get("http_host") or "127.0.0.1").strip()
		return f"HTTP 已启用（streamable-http）：http://{host}:{port}"
	return "HTTP 未启用。可在 model.conf 设置 http_port=145387，http_host=0.0.0.0。（示例）"


@app.tool(
	name="ingest_urls",
	description="Fetch visible text from URLs, chunk, and add to the Chroma vector store.",
	tags={"ingest", "web", "docs"},
)  # type: ignore[union-attr]
def ingest_urls(urls: Annotated[List[str], Field(description="List of URLs to fetch and index")]) -> str:
	"""Fetch visible text from URLs and add to the Chroma vector store. Scripts/styles are removed."""
	ensure_dirs()
	conf = load_conf()
	# Use LangChain vectorstore to add texts consistently
	if LCChroma is None:
		raise RuntimeError("LangChain 未安装")
	lc_embeddings = _make_lc_embeddings(conf)
	vectorstore = LCChroma(collection_name="docs", persist_directory=DB_DIR, embedding_function=lc_embeddings)
	import requests
	from bs4 import BeautifulSoup  # type: ignore

	added = 0
	for url in urls:
		try:
			r = requests.get(url, timeout=30)
			r.raise_for_status()
			soup = BeautifulSoup(r.text, "html.parser")
			# crude text extraction
			for s in soup(["script", "style", "noscript"]):
				s.extract()
			text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
			chunks = chunk_text(text)
			if not chunks:
				continue
			ids = [f"{url}::chunk-{i}" for i in range(len(chunks))]
			metas = [{"source_url": url, "chunk_index": i} for i in range(len(chunks))]
			vectorstore.add_texts(texts=chunks, metadatas=cast(Any, metas), ids=ids)
			added += len(chunks)
		except Exception:
			continue
	# Chroma auto-persists; skip manual persist
	return f"已从 {len(urls)} 个 URL 抓取并加入 {added} 个文本分片。"


if __name__ == "__main__":
	# 启动 FastMCP 服务器：
	# - 默认使用 STDIO（适配 VS Code/Claude 等本地 MCP 客户端）
	# - 如果 model.conf 配置了 http_port，则以 HTTP 方式在该端口启动
	ensure_dirs()
	if FastMCP is None:
		print("FastMCP 未安装，无法启动。请先安装 fastmcp 依赖。", file=sys.stderr)
		raise SystemExit(1)
	conf = load_conf()
	port = int(conf.get("http_port", 0) or 0)
	if port > 0:
		# HTTP 模式（使用 streamable-http）
		host = str(conf.get("http_host") or "127.0.0.1").strip()
		app.run(transport="streamable-http", host=host, port=port)
	else:
		# STDIO 模式（默认）
		app.run()
