"""Private cooking knowledge-base ingestion and retrieval."""

from __future__ import annotations

import hashlib
import io
import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


ALLOWED_KNOWLEDGE_TYPES = {
    ".pdf": {"application/pdf", "application/octet-stream"},
    ".txt": {"text/plain", "application/octet-stream"},
    ".md": {"text/markdown", "text/plain", "application/octet-stream"},
}


class KnowledgeError(RuntimeError):
    """Base error for public knowledge-base operations."""


class KnowledgeUnavailableError(KnowledgeError):
    """Raised when the embedding/vector service is unavailable."""


class KnowledgeValidationError(KnowledgeError):
    """Raised when an uploaded knowledge document is invalid."""


def _safe_filename(filename: str) -> str:
    name = Path(filename.replace("\\", "/")).name.strip()
    name = re.sub(r"[\x00-\x1f]", "", name)
    return name[:180] or "knowledge-document"


class KnowledgeService:
    """Owns document metadata, original files, vector indexing, and search."""

    def __init__(
        self,
        root: Path,
        *,
        vector_store: Any | None = None,
        embeddings: Any | None = None,
        initialization_error: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.data_dir = self.root / "data"
        self.files_dir = self.data_dir / "knowledge_files"
        self.chroma_dir = self.data_dir / "chroma"
        self.data_dir.mkdir(exist_ok=True)
        self.files_dir.mkdir(exist_ok=True)

        self.top_k = max(1, min(int(os.getenv("RAG_TOP_K", "6")), 12))
        self.score_threshold = min(
            1.0, max(0.0, float(os.getenv("RAG_SCORE_THRESHOLD", "0.35")))
        )
        self.max_file_bytes = max(
            1, int(os.getenv("MAX_KNOWLEDGE_FILE_MB", "10"))
        ) * 1024 * 1024
        self.initialization_error = initialization_error
        self._lock = threading.RLock()
        self._db = sqlite3.connect(
            self.data_dir / "knowledge.sqlite", check_same_thread=False
        )
        self._db.row_factory = sqlite3.Row
        self._create_schema()

        if vector_store is not None:
            self.vector_store = vector_store
        elif embeddings is not None:
            from langchain_chroma import Chroma

            self.chroma_dir.mkdir(exist_ok=True)
            self.vector_store = Chroma(
                collection_name="ai_private_chef_knowledge_v3",
                embedding_function=embeddings,
                persist_directory=str(self.chroma_dir),
            )
        else:
            self.vector_store = None

    @classmethod
    def from_environment(cls, root: Path) -> "KnowledgeService":
        api_key = os.getenv("QWEN_API_KEY")
        if not api_key:
            return cls(
                root,
                initialization_error="未配置通义千问 API Key，知识库向量化不可用。",
            )
        try:
            dimensions = int(os.getenv("QWEN_EMBEDDING_DIMENSIONS", "1024"))
            embeddings = OpenAIEmbeddings(
                model=os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v4"),
                dimensions=dimensions,
                api_key=api_key,
                base_url=os.getenv(
                    "QWEN_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
                chunk_size=10,
                max_retries=2,
                timeout=60,
                check_embedding_ctx_length=False,
            )
            return cls(root, embeddings=embeddings)
        except Exception:
            return cls(
                root,
                initialization_error="知识库向量服务初始化失败，请检查 Embedding 配置。",
            )

    @property
    def ready(self) -> bool:
        return self.vector_store is not None and self.initialization_error is None

    def _create_schema(self) -> None:
        with self._db:
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    stored_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    chunk_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    document_id TEXT NOT NULL,
                    chunk_id TEXT PRIMARY KEY,
                    page INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                )
                """
            )

    def _require_ready(self) -> None:
        if not self.ready:
            raise KnowledgeUnavailableError(
                self.initialization_error
                or "知识库暂不可用，请检查通义千问 Embedding 配置。"
            )

    def _validate_upload(self, filename: str, content_type: str, data: bytes) -> tuple[str, str]:
        safe_name = _safe_filename(filename)
        suffix = Path(safe_name).suffix.lower()
        if suffix not in ALLOWED_KNOWLEDGE_TYPES:
            raise KnowledgeValidationError("仅支持 PDF、TXT 和 Markdown 文件。")
        normalized_type = (content_type or "application/octet-stream").split(";", 1)[0].lower()
        if normalized_type not in ALLOWED_KNOWLEDGE_TYPES[suffix]:
            raise KnowledgeValidationError("文件内容类型与扩展名不匹配。")
        if not data:
            raise KnowledgeValidationError("知识文件不能为空。")
        if len(data) > self.max_file_bytes:
            limit_mb = self.max_file_bytes // (1024 * 1024)
            raise KnowledgeValidationError(f"知识文件不能超过 {limit_mb} MB。")
        return safe_name, suffix

    @staticmethod
    def _decode_text(data: bytes) -> str:
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise KnowledgeValidationError("文本编码无法识别，请保存为 UTF-8 后重试。")

    def _extract_pages(self, suffix: str, data: bytes) -> list[tuple[int, str]]:
        if suffix != ".pdf":
            text = self._decode_text(data).strip()
            if not text:
                raise KnowledgeValidationError("文件中没有可索引的文字。")
            return [(1, text)]

        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            pages = []
            for number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append((number, text))
        except KnowledgeValidationError:
            raise
        except Exception as exc:
            raise KnowledgeValidationError("PDF 解析失败，请确认文件未损坏或加密。") from exc
        if not pages:
            raise KnowledgeValidationError(
                "PDF 中没有可提取文字；扫描版 PDF 暂不支持 OCR。"
            )
        return pages

    def ingest_document(
        self, filename: str, content_type: str, data: bytes
    ) -> dict[str, Any]:
        self._require_ready()
        safe_name, suffix = self._validate_upload(filename, content_type, data)
        file_hash = hashlib.sha256(data).hexdigest()

        with self._lock:
            existing = self._db.execute(
                "SELECT * FROM documents WHERE sha256 = ?", (file_hash,)
            ).fetchone()
            if existing:
                result = self._row_to_document(existing)
                result["duplicate"] = True
                return result

        pages = self._extract_pages(suffix, data)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )
        document_id = uuid.uuid4().hex
        documents: list[Document] = []
        chunk_rows: list[tuple[str, str, int, int]] = []
        chunk_ids: list[str] = []

        for page_number, page_text in pages:
            for chunk_index, text in enumerate(splitter.split_text(page_text)):
                cleaned = text.strip()
                if not cleaned:
                    continue
                chunk_id = f"{document_id}:{page_number}:{chunk_index}"
                documents.append(
                    Document(
                        page_content=cleaned,
                        metadata={
                            "document_id": document_id,
                            "filename": safe_name,
                            "file_type": suffix.lstrip("."),
                            "page": page_number,
                            "chunk_index": chunk_index,
                        },
                    )
                )
                chunk_ids.append(chunk_id)
                chunk_rows.append((document_id, chunk_id, page_number, chunk_index))

        if not documents:
            raise KnowledgeValidationError("文件中没有可索引的有效文字。")

        stored_name = f"{document_id}{suffix}"
        stored_path = self.files_dir / stored_name
        vectors_added = False
        try:
            self.vector_store.add_documents(documents=documents, ids=chunk_ids)
            vectors_added = True
            stored_path.write_bytes(data)
            created_at = datetime.now().astimezone().isoformat(timespec="seconds")
            with self._lock, self._db:
                self._db.execute(
                    """
                    INSERT INTO documents
                    (id, filename, stored_name, file_type, content_type, size_bytes,
                     sha256, chunk_count, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        safe_name,
                        stored_name,
                        suffix.lstrip("."),
                        content_type or "application/octet-stream",
                        len(data),
                        file_hash,
                        len(documents),
                        created_at,
                    ),
                )
                self._db.executemany(
                    "INSERT INTO chunks (document_id, chunk_id, page, chunk_index) VALUES (?, ?, ?, ?)",
                    chunk_rows,
                )
            return self.get_document(document_id)
        except KnowledgeError:
            raise
        except Exception as exc:
            if vectors_added:
                try:
                    self.vector_store.delete(ids=chunk_ids)
                except Exception:
                    pass
            stored_path.unlink(missing_ok=True)
            error_text = str(exc).lower()
            if any(token in error_text for token in ("403", "401", "access denied", "api key", "model not")):
                self.initialization_error = (
                    "Embedding 模型访问被拒绝，请为 API Key 授权 text-embedding-v4。"
                )
            raise KnowledgeError(
                "文档向量化失败，请检查 text-embedding-v4 权限和 API 配置。"
            ) from exc

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "filename": row["filename"],
            "file_type": row["file_type"],
            "content_type": row["content_type"],
            "size_bytes": row["size_bytes"],
            "chunk_count": row["chunk_count"],
            "status": "ready",
            "created_at": row["created_at"],
        }

    def list_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def get_document(self, document_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        if not row:
            raise KeyError(document_id)
        return self._row_to_document(row)

    def get_download(self, document_id: str) -> tuple[Path, str, str]:
        with self._lock:
            row = self._db.execute(
                "SELECT filename, stored_name, content_type FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
        if not row:
            raise KeyError(document_id)
        path = self.files_dir / row["stored_name"]
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, row["filename"], row["content_type"]

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            row = self._db.execute(
                "SELECT stored_name FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if not row:
                return False
            chunk_rows = self._db.execute(
                "SELECT chunk_id FROM chunks WHERE document_id = ?", (document_id,)
            ).fetchall()
            chunk_ids = [item["chunk_id"] for item in chunk_rows]

        if self.ready and chunk_ids:
            self.vector_store.delete(ids=chunk_ids)
        with self._lock, self._db:
            self._db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            self._db.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        (self.files_dir / row["stored_name"]).unlink(missing_ok=True)
        return True

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        self._require_ready()
        cleaned_query = query.strip()
        if not cleaned_query:
            raise KnowledgeValidationError("检索问题不能为空。")
        final_k = max(1, min(top_k or self.top_k, 12))
        fetch_k = max(12, final_k)
        try:
            try:
                matches = self.vector_store.similarity_search_with_relevance_scores(
                    cleaned_query, k=fetch_k
                )
            except NotImplementedError:
                # Some test/local stores already return cosine similarity directly
                # but do not implement LangChain's relevance-score adapter.
                matches = self.vector_store.similarity_search_with_score(
                    cleaned_query, k=fetch_k
                )
        except Exception as exc:
            raise KnowledgeError("知识库检索失败，请稍后重试。") from exc

        results = []
        with self._lock:
            valid_document_ids = {
                row["id"]
                for row in self._db.execute("SELECT id FROM documents").fetchall()
            }
        for document, raw_score in matches:
            score = max(0.0, min(1.0, float(raw_score)))
            if score < self.score_threshold:
                continue
            metadata = document.metadata
            document_id = str(metadata.get("document_id", ""))
            if document_id not in valid_document_ids:
                continue
            results.append(
                {
                    "document_id": document_id,
                    "title": str(metadata.get("filename", "知识库文档")),
                    "page": int(metadata.get("page", 1)),
                    "chunk_index": int(metadata.get("chunk_index", 0)),
                    "score": round(score, 4),
                    "content": document.page_content,
                    "url": f"/api/knowledge/documents/{document_id}/download",
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:final_k]

    def stats(self) -> dict[str, int]:
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) AS documents, COALESCE(SUM(chunk_count), 0) AS chunks FROM documents"
            ).fetchone()
        return {"documents": int(row["documents"]), "chunks": int(row["chunks"])}

    def seed_sample_documents(self, sample_dir: Path) -> int:
        """Index bundled, non-sensitive Markdown samples once."""
        if not self.ready or not sample_dir.is_dir():
            return 0
        indexed = 0
        for path in sorted(sample_dir.glob("*.md")):
            result = self.ingest_document(
                path.name,
                "text/markdown",
                path.read_bytes(),
            )
            if not result.get("duplicate"):
                indexed += 1
        return indexed

    def close(self) -> None:
        with self._lock:
            self._db.close()
