from fastapi.testclient import TestClient

from app.main import app, content_to_text


client = TestClient(app)


def test_home_page_is_available():
    response = client.get("/")
    assert response.status_code == 200
    assert "AI 私厨" in response.text


def test_health_does_not_expose_secrets():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    serialized = response.text.lower()
    assert "api_key" not in serialized
    assert "access_key" not in serialized


def test_rejects_non_image_upload():
    response = client.post(
        "/api/upload",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_rejects_invalid_thread_id():
    response = client.get("/api/history/invalid%20thread")
    assert response.status_code == 422


def test_stream_text_preserves_markdown_whitespace():
    assert content_to_text("\n## 推荐结果\n", strip=False) == "\n## 推荐结果\n"
