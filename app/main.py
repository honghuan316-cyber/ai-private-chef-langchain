import json
import logging
import os
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessageChunk, ToolMessage
from pydantic import BaseModel, Field, field_validator

from app.chef_core import get_agent


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

app = FastAPI(title="AI 私厨", version="2.0.0")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS)), name="uploads")

agent, checkpointer = get_agent()


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=4000)
    image_url: str | None = Field(default=None, max_length=5000)
    thread_id: str = Field(min_length=1, max_length=100)

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


def make_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def make_message(request: ChatRequest) -> dict:
    message = request.message
    if not message and request.image_url:
        message = "请识别图片中的食材，并推荐两道适合的菜。"
    if not message:
        raise HTTPException(status_code=422, detail="请输入需求或上传食材图片")
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
    message = str(exc).lower()
    if "api key" in message or "access denied" in message or "403" in message:
        return "模型访问被拒绝，请检查 API Key 和模型权限。"
    if "timeout" in message or "timed out" in message:
        return "模型响应超时，请稍后重试。"
    if "tavily" in message:
        return "菜谱搜索暂时不可用，请检查 Tavily 配置。"
    return "AI 服务暂时不可用，请稍后重试。"


@app.get("/")
def index():
    return FileResponse(
        ROOT / "static" / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/health")
def health():
    qwen_ready = bool(os.getenv("QWEN_API_KEY"))
    tavily_ready = bool(os.getenv("TAVILY_API_KEY"))
    oss_ready = all(
        os.getenv(name)
        for name in ("OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET", "OSS_ENDPOINT", "OSS_BUCKET")
    )
    ready = qwen_ready and tavily_ready
    return {
        "status": "ok" if ready else "degraded",
        "version": app.version,
        "model": os.getenv("QWEN_MODEL", "未配置"),
        "services": {"qwen": qwen_ready, "tavily": tavily_ready, "oss": oss_ready},
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="仅支持 JPG、PNG 或 WebP 图片")

    data = await file.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="图片文件为空")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="图片不能超过 8 MB")

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
            bucket.put_object(key, data, headers={"Content-Type": file.content_type})
            return {
                "image_url": bucket.sign_url("GET", key, 900),
                "storage": "oss",
                "expires_in": 900,
            }
        except Exception:
            logger.exception("OSS upload failed")
            raise HTTPException(status_code=502, detail="图片上传失败，请检查 OSS 配置")

    local_name = key.replace("/", "-")
    (UPLOADS / local_name).write_bytes(data)
    return {
        "image_url": f"/uploads/{local_name}",
        "storage": "local",
        "warning": "当前使用本地临时存储",
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        result = agent.invoke(
            {"messages": [make_message(request)]},
            config=make_config(request.thread_id),
        )
        return {
            "thread_id": request.thread_id,
            "answer": content_to_text(result["messages"][-1].content),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=502, detail=public_error_message(exc)) from exc


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest):
    user_message = make_message(request)

    def events():
        yield f"data: {json.dumps({'type': 'status', 'message': '正在分析你的需求…'}, ensure_ascii=False)}\n\n"
        search_status_sent = False
        try:
            for chunk, _metadata in agent.stream(
                {"messages": [user_message]},
                config=make_config(request.thread_id),
                stream_mode="messages",
            ):
                if isinstance(chunk, ToolMessage):
                    if not search_status_sent:
                        search_status_sent = True
                        yield f"data: {json.dumps({'type': 'status', 'message': '已找到菜谱资料，正在评分整理…'}, ensure_ascii=False)}\n\n"
                    continue
                if not isinstance(chunk, AIMessageChunk):
                    continue
                if getattr(chunk, "tool_call_chunks", None):
                    if not search_status_sent:
                        search_status_sent = True
                        yield f"data: {json.dumps({'type': 'status', 'message': '正在使用 Tavily 搜索可靠菜谱…'}, ensure_ascii=False)}\n\n"
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
def history(thread_id: str):
    if not THREAD_ID_PATTERN.fullmatch(thread_id):
        raise HTTPException(status_code=422, detail="无效的会话编号")
    state = agent.get_state(make_config(thread_id))
    messages = []
    for message in state.values.get("messages", []):
        role = getattr(message, "type", "")
        if role not in ("human", "ai"):
            continue
        if role == "ai" and getattr(message, "tool_calls", None):
            continue
        content = content_to_text(getattr(message, "content", ""))
        if content:
            messages.append({
                "role": "user" if role == "human" else "assistant",
                "content": content,
            })
    return {"thread_id": thread_id, "messages": messages}


@app.delete("/api/history/{thread_id}")
def clear_history(thread_id: str):
    if not THREAD_ID_PATTERN.fullmatch(thread_id):
        raise HTTPException(status_code=422, detail="无效的会话编号")
    checkpointer.delete_thread(thread_id)
    return {"thread_id": thread_id, "cleared": True}
