"""FastAPI application for the multimodal AI Private Chef."""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessageChunk, ToolMessage
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from app.chef_core import ChefRuntime
from app.knowledge import (
    KnowledgeError,
    KnowledgeService,
    KnowledgeUnavailableError,
    KnowledgeValidationError,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai-private-chef")

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "uploads"
UPLOADS.mkdir(exist_ok=True)

MAX_IMAGE_BYTES = 8 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
DOCUMENT_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


@asynccontextmanager
async def lifespan(application: FastAPI):
    knowledge = KnowledgeService.from_environment(ROOT)
    if (
        knowledge.ready
        and os.getenv("AUTO_INDEX_SAMPLE_KNOWLEDGE", "true").lower() in {"1", "true", "yes"}
        and knowledge.stats()["documents"] == 0
    ):
        try:
            await run_in_threadpool(
                knowledge.seed_sample_documents, ROOT / "knowledge_samples"
            )
        except KnowledgeError:
            logger.exception("Bundled knowledge samples could not be indexed")
    runtime = ChefRuntime(ROOT, knowledge)
    application.state.knowledge = knowledge
    application.state.runtime = runtime
    try:
        yield
    finally:
        runtime.close()
        knowledge.close()


app = FastAPI(title="AI 私厨", version="3.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS)), name="uploads")


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=4000)
    image_url: str | None = Field(default=None, max_length=5000)
    thread_id: str = Field(min_length=1, max_length=100)
    source_mode: Literal["auto", "knowledge", "web"] = "auto"

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        return value.strip()

    @field_validator("thread_id")
    @classmethod
    def validate_thread_id(cls, value: str) -> str:
        if not THREAD_ID_PATTERN.fullmatch(value):
            raise ValueError("thread_id 只能包含字母、数字、下划线和连字符")
        return value


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=6, ge=1, le=12)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("检索问题不能为空")
        return value


def get_knowledge(request: Request) -> KnowledgeService:
    knowledge = getattr(request.app.state, "knowledge", None)
    if knowledge is None:
        raise HTTPException(status_code=503, detail="知识库服务尚未启动。")
    return knowledge


def get_runtime(request: Request) -> ChefRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None or not runtime.ready:
        detail = getattr(runtime, "initialization_error", None) or "AI Agent 尚未就绪。"
        raise HTTPException(status_code=503, detail=detail)
    return runtime


def validate_document_id(document_id: str) -> str:
    if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
        raise HTTPException(status_code=422, detail="无效的知识文档编号。")
    return document_id


def make_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def make_message(request: ChatRequest) -> dict:
    message = request.message
    if not message and request.image_url:
        message = "请识别图片中的食材，并推荐两道适合的菜。"
    if not message:
        raise HTTPException(status_code=422, detail="请输入需求或上传食材图片。")
    if not request.image_url:
        return {"role": "user", "content": message}
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": message},
            {"type": "image_url", "image_url": {"url": request.image_url}},
        ],
    }


def content_to_text(content, *, strip: bool = True) -> str:
    if isinstance(content, str):
        return content.strip() if strip else content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                parts.append(str(block.get("text", "")))
        text = "\n".join(part for part in parts if part)
        return text.strip() if strip else text
    return ""


def public_error_message(exc: Exception) -> str:
    if isinstance(exc, KnowledgeError):
        return str(exc)
    message = str(exc).lower()
    if "api key" in message or "access denied" in message or "403" in message:
        return "模型访问被拒绝，请检查 API Key 和模型权限。"
    if "timeout" in message or "timed out" in message:
        return "模型响应超时，请稍后重试。"
    if "tavily" in message:
        return "网络菜谱搜索暂时不可用，请检查 Tavily 配置。"
    if "embedding" in message:
        return "知识库向量服务不可用，请检查 text-embedding-v4 权限。"
    return "AI 服务暂时不可用，请稍后重试。"


@app.get("/")
def index():
    return FileResponse(
        ROOT / "static" / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/health")
def health(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    knowledge = getattr(request.app.state, "knowledge", None)
    qwen_ready = bool(runtime and runtime.ready)
    tavily_ready = bool(os.getenv("TAVILY_API_KEY"))
    oss_ready = all(
        os.getenv(name)
        for name in ("OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET", "OSS_ENDPOINT", "OSS_BUCKET")
    )
    rag_ready = bool(knowledge and knowledge.ready)
    ready = qwen_ready and tavily_ready
    return {
        "status": "ok" if ready else "degraded",
        "version": app.version,
        "model": os.getenv("QWEN_MODEL", "未配置"),
        "embedding_model": os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v4"),
        "services": {
            "qwen": qwen_ready,
            "tavily": tavily_ready,
            "oss": oss_ready,
            "rag": rag_ready,
        },
        "knowledge": knowledge.stats() if knowledge else {"documents": 0, "chunks": 0},
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="仅支持 JPG、PNG 或 WebP 图片。")

    data = await file.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="图片文件为空。")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="图片不能超过 8 MB。")

    suffix = ALLOWED_IMAGE_TYPES[file.content_type]
    key = f"ingredients/{uuid.uuid4().hex}{suffix}"
    oss_values = {
        "id": os.getenv("OSS_ACCESS_KEY_ID"),
        "secret": os.getenv("OSS_ACCESS_KEY_SECRET"),
        "endpoint": os.getenv("OSS_ENDPOINT"),
        "bucket": os.getenv("OSS_BUCKET"),
    }

    if all(oss_values.values()):
        try:
            import oss2

            bucket = oss2.Bucket(
                oss2.Auth(oss_values["id"], oss_values["secret"]),
                oss_values["endpoint"],
                oss_values["bucket"],
            )
            await run_in_threadpool(
                bucket.put_object, key, data, {"Content-Type": file.content_type}
            )
            return {
                "image_url": bucket.sign_url("GET", key, 900),
                "storage": "oss",
                "expires_in": 900,
            }
        except Exception as exc:
            logger.exception("OSS upload failed")
            raise HTTPException(
                status_code=502, detail="图片上传失败，请检查 OSS 配置。"
            ) from exc

    local_name = key.replace("/", "-")
    (UPLOADS / local_name).write_bytes(data)
    return {
        "image_url": f"/uploads/{local_name}",
        "storage": "local",
        "warning": "当前使用本地临时存储。",
    }


@app.post("/api/knowledge/documents", status_code=status.HTTP_201_CREATED)
async def upload_knowledge_document(request: Request, file: UploadFile = File(...)):
    knowledge = get_knowledge(request)
    data = await file.read(knowledge.max_file_bytes + 1)
    try:
        return await run_in_threadpool(
            knowledge.ingest_document,
            file.filename or "knowledge-document",
            file.content_type or "application/octet-stream",
            data,
        )
    except KnowledgeValidationError as exc:
        message = str(exc)
        code = 413 if "不能超过" in message else 400
        raise HTTPException(status_code=code, detail=message) from exc
    except KnowledgeUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KnowledgeError as exc:
        logger.exception("Knowledge ingestion failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/knowledge/documents")
def list_knowledge_documents(request: Request):
    knowledge = get_knowledge(request)
    stats = knowledge.stats()
    return {
        "documents": knowledge.list_documents(),
        "document_count": stats["documents"],
        "chunks": stats["chunks"],
    }


@app.delete("/api/knowledge/documents/{document_id}")
def delete_knowledge_document(document_id: str, request: Request):
    validate_document_id(document_id)
    knowledge = get_knowledge(request)
    try:
        deleted = knowledge.delete_document(document_id)
    except KnowledgeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="知识文档不存在。")
    return {"id": document_id, "deleted": True}


@app.get("/api/knowledge/documents/{document_id}/download")
def download_knowledge_document(document_id: str, request: Request):
    validate_document_id(document_id)
    knowledge = get_knowledge(request)
    try:
        path, filename, content_type = knowledge.get_download(document_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="知识文档不存在。") from exc
    return FileResponse(path, media_type=content_type, filename=filename)


@app.post("/api/knowledge/search")
async def search_knowledge(payload: KnowledgeSearchRequest, request: Request):
    knowledge = get_knowledge(request)
    try:
        matches = await run_in_threadpool(knowledge.search, payload.query, payload.top_k)
        return {"query": payload.query, "matches": matches, "count": len(matches)}
    except KnowledgeUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KnowledgeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/chat")
def chat(request_body: ChatRequest, request: Request):
    runtime = get_runtime(request)
    try:
        result = runtime.agent.invoke(
            {"messages": [make_message(request_body)]},
            config=make_config(request_body.thread_id),
            source_mode=request_body.source_mode,
        )
        return {
            "thread_id": request_body.thread_id,
            "answer": content_to_text(result["messages"][-1].content),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=502, detail=public_error_message(exc)) from exc


def _new_sources(artifact, seen: set[tuple]) -> list[dict]:
    if not isinstance(artifact, dict):
        return []
    unique = []
    for source in artifact.get("sources", []):
        if not isinstance(source, dict):
            continue
        key = (
            source.get("source_type"),
            source.get("url"),
            source.get("document_id"),
            source.get("page"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


@app.post("/api/chat/stream")
def chat_stream(request_body: ChatRequest, request: Request):
    runtime = get_runtime(request)
    user_message = make_message(request_body)

    def events():
        yield f"data: {json.dumps({'type': 'status', 'message': '正在分析你的需求…'}, ensure_ascii=False)}\n\n"
        seen_sources: set[tuple] = set()
        try:
            for chunk, _metadata in runtime.agent.stream(
                {"messages": [user_message]},
                config=make_config(request_body.thread_id),
                stream_mode="messages",
                source_mode=request_body.source_mode,
            ):
                if isinstance(chunk, ToolMessage):
                    sources = _new_sources(getattr(chunk, "artifact", None), seen_sources)
                    if sources:
                        yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"
                    tool_name = getattr(chunk, "name", "") or ""
                    if tool_name == "search_private_knowledge":
                        message = "知识库检索完成，正在整理资料…"
                    elif tool_name == "search_web_recipes":
                        message = "网络菜谱搜索完成，正在评分排序…"
                    else:
                        message = "资料检索完成，正在生成回答…"
                    yield f"data: {json.dumps({'type': 'status', 'message': message}, ensure_ascii=False)}\n\n"
                    continue
                if not isinstance(chunk, AIMessageChunk):
                    continue
                tool_calls = getattr(chunk, "tool_call_chunks", None) or []
                if tool_calls:
                    names = {item.get("name") for item in tool_calls if isinstance(item, dict)}
                    if "search_private_knowledge" in names:
                        message = "正在检索私人烹饪知识库…"
                    elif "search_web_recipes" in names:
                        message = "正在使用 Tavily 搜索网络菜谱…"
                    else:
                        message = "正在调用检索工具…"
                    yield f"data: {json.dumps({'type': 'status', 'message': message}, ensure_ascii=False)}\n\n"
                    continue
                content = content_to_text(chunk.content, strip=False)
                if content:
                    yield f"data: {json.dumps({'type': 'text', 'text': content}, ensure_ascii=False)}\n\n"
            yield 'data: {"type":"done"}\n\n'
        except Exception as exc:
            logger.exception("Streaming chat failed")
            error = public_error_message(exc)
            yield f"data: {json.dumps({'type': 'error', 'message': error}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/history/{thread_id}")
def history(thread_id: str, request: Request):
    if not THREAD_ID_PATTERN.fullmatch(thread_id):
        raise HTTPException(status_code=422, detail="无效的会话编号。")
    runtime = get_runtime(request)
    state = runtime.agent.get_state(make_config(thread_id))
    messages = []
    for message in state.values.get("messages", []):
        role = getattr(message, "type", "")
        if role not in ("human", "ai"):
            continue
        if role == "ai" and getattr(message, "tool_calls", None):
            continue
        content = content_to_text(getattr(message, "content", ""))
        if content:
            messages.append(
                {"role": "user" if role == "human" else "assistant", "content": content}
            )
    return {"thread_id": thread_id, "messages": messages}


@app.delete("/api/history/{thread_id}")
def clear_history(thread_id: str, request: Request):
    if not THREAD_ID_PATTERN.fullmatch(thread_id):
        raise HTTPException(status_code=422, detail="无效的会话编号。")
    runtime = get_runtime(request)
    runtime.checkpointer.delete_thread(thread_id)
    return {"thread_id": thread_id, "cleared": True}
