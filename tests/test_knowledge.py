from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore

from app.knowledge import KnowledgeService, KnowledgeValidationError


@pytest.fixture()
def knowledge(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RAG_SCORE_THRESHOLD", "0")
    embeddings = DeterministicFakeEmbedding(size=64)
    vector_store = InMemoryVectorStore(embeddings)
    service = KnowledgeService(tmp_path, vector_store=vector_store)
    try:
        yield service
    finally:
        service.close()


def test_text_ingestion_deduplication_search_and_delete(knowledge):
    content = "番茄富含多种营养成分。鸡蛋可以提供优质蛋白质。减脂晚餐应注意少油少盐。"
    first = knowledge.ingest_document("家庭营养.txt", "text/plain", content.encode("utf-8"))
    assert first["status"] == "ready"
    assert first["chunk_count"] >= 1

    duplicate = knowledge.ingest_document("副本.txt", "text/plain", content.encode("utf-8"))
    assert duplicate["id"] == first["id"]
    assert duplicate["duplicate"] is True
    assert knowledge.stats()["documents"] == 1

    matches = knowledge.search("减脂晚餐", top_k=5)
    assert matches
    assert matches[0]["title"] == "家庭营养.txt"
    assert matches[0]["page"] == 1

    assert knowledge.delete_document(first["id"]) is True
    assert knowledge.stats() == {"documents": 0, "chunks": 0}
    assert knowledge.delete_document(first["id"]) is False


def test_markdown_ingestion(knowledge):
    result = knowledge.ingest_document(
        "菜谱.md", "text/markdown", "# 清蒸鱼\n\n使用姜葱和少量盐清蒸。".encode("utf-8")
    )
    assert result["file_type"] == "md"
    assert result["chunk_count"] == 1


def test_pdf_ingestion_preserves_page_number(knowledge, monkeypatch):
    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakeReader:
        def __init__(self, _stream):
            self.pages = [FakePage("第一页介绍食材。"), FakePage("第二页说明少油烹饪。")]

    import pypdf

    monkeypatch.setattr(pypdf, "PdfReader", FakeReader)
    result = knowledge.ingest_document("指南.pdf", "application/pdf", b"%PDF-test")
    assert result["chunk_count"] == 2
    pages = {item["page"] for item in knowledge.search("少油烹饪", top_k=5)}
    assert pages.issubset({1, 2})


def test_scanned_pdf_without_text_is_rejected(knowledge, monkeypatch):
    class EmptyPage:
        def extract_text(self):
            return ""

    class EmptyReader:
        def __init__(self, _stream):
            self.pages = [EmptyPage()]

    import pypdf

    monkeypatch.setattr(pypdf, "PdfReader", EmptyReader)
    with pytest.raises(KnowledgeValidationError, match="OCR"):
        knowledge.ingest_document("扫描件.pdf", "application/pdf", b"%PDF-empty")


@pytest.mark.parametrize(
    ("filename", "content_type", "data"),
    [
        ("空文件.txt", "text/plain", b""),
        ("资料.docx", "application/octet-stream", b"content"),
        ("伪装.pdf", "text/plain", b"content"),
    ],
)
def test_invalid_documents_are_rejected(knowledge, filename, content_type, data):
    with pytest.raises(KnowledgeValidationError):
        knowledge.ingest_document(filename, content_type, data)
