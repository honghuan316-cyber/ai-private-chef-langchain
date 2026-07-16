import os

os.environ["AUTO_INDEX_SAMPLE_KNOWLEDGE"] = "false"

import pytest
from fastapi.testclient import TestClient

from app.main import _new_sources, app, content_to_text


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_home_page_is_available(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "AI 私厨" in response.text
    assert "知识库" in response.text


def test_health_does_not_expose_secrets(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert "rag" in body["services"]
    serialized = response.text.lower()
    assert "api_key" not in serialized
    assert "access_key" not in serialized
    assert os.getenv("QWEN_API_KEY", "not-configured") not in response.text


def test_rejects_non_image_upload(client):
    response = client.post(
        "/api/upload",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_rejects_unsupported_knowledge_file(client):
    response = client.post(
        "/api/knowledge/documents",
        files={"file": ("notes.docx", b"not a document", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_knowledge_list_keeps_documents_and_counts_separate(client):
    response = client.get("/api/knowledge/documents")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["documents"], list)
    assert isinstance(body["document_count"], int)
    assert isinstance(body["chunks"], int)


def test_rejects_empty_knowledge_search(client):
    response = client.post("/api/knowledge/search", json={"query": "", "top_k": 6})
    assert response.status_code == 422


def test_rejects_invalid_thread_id(client):
    response = client.get("/api/history/invalid%20thread")
    assert response.status_code == 422


def test_stream_text_preserves_markdown_whitespace():
    assert content_to_text("\n## 推荐结果\n", strip=False) == "\n## 推荐结果\n"


def test_source_events_are_deduplicated():
    source = {
        "source_type": "knowledge",
        "title": "营养资料.md",
        "document_id": "a" * 32,
        "page": 2,
        "score": 0.8,
        "url": "/download",
    }
    seen = set()
    assert _new_sources({"sources": [source]}, seen) == [source]
    assert _new_sources({"sources": [source]}, seen) == []
