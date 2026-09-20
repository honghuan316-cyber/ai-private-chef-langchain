import os

os.environ["AUTO_INDEX_SAMPLE_KNOWLEDGE"] = "false"

import pytest
from fastapi.testclient import TestClient

from app.chef_core import PolicyRoutedAgent
from app.knowledge import KnowledgeUnavailableError
from app.main import ChatRequest, app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_rejects_invalid_source_mode(client):
    response = client.post(
        "/api/chat/stream",
        json={"message": "推荐晚餐", "thread_id": "test-thread", "source_mode": "invalid"},
    )
    assert response.status_code == 422


def test_source_mode_defaults_to_auto():
    request = ChatRequest(message="推荐晚餐", thread_id="test-thread")

    assert request.source_mode == "auto"


def test_source_mode_forces_selected_tool():
    class NamedTool:
        def __init__(self, name):
            self.name = name

    router = PolicyRoutedAgent(
        graph=None,
        tools=[NamedTool("search_private_knowledge"), NamedTool("search_web_recipes")],
    )

    assert router._required_tools("推荐两道菜", "knowledge") == [
        "search_private_knowledge"
    ]
    assert router._required_tools("推荐两道菜", "web") == ["search_web_recipes"]
    assert router._required_tools("普通问候", "auto") == []


def test_forced_source_mode_reports_unavailable_tool():
    router = PolicyRoutedAgent(graph=None, tools=[])

    with pytest.raises(KnowledgeUnavailableError, match="暂不可用"):
        router._required_tools("推荐两道菜", "knowledge")
    with pytest.raises(RuntimeError, match="Tavily"):
        router._required_tools("推荐两道菜", "web")


def test_routing_extracts_text_from_multimodal_message():
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "根据知识库推荐晚餐"},
            {"type": "image_url", "image_url": {"url": "/uploads/example.jpg"}},
        ],
    }

    assert PolicyRoutedAgent._query_from_input({"messages": [message]}) == (
        "根据知识库推荐晚餐"
    )
