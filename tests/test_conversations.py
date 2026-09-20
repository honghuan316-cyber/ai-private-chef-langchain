from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from app.chef_core import ChefRuntime
from app.conversations import (
    ConversationStore,
    clean_generated_title,
    conversation_summaries,
)
from app.main import ConversationTitleRequest, rename_conversation_title


class FakeCheckpointer:
    def __init__(self, items):
        self.items = items

    def list(self, _config):
        return iter(self.items)


def checkpoint(thread_id, timestamp, messages):
    return SimpleNamespace(
        config={"configurable": {"thread_id": thread_id}},
        checkpoint={"ts": timestamp, "channel_values": {"messages": messages}},
    )


def test_conversation_summaries_use_latest_checkpoint_and_saved_title(tmp_path):
    first_thread = "11111111-1111-4111-8111-111111111111"
    second_thread = "22222222-2222-4222-8222-222222222222"
    store = ConversationStore(tmp_path / "conversations.sqlite")
    store.save_title(first_thread, "鸡蛋番茄减脂午餐")
    saver = FakeCheckpointer(
        [
            checkpoint(
                first_thread,
                "2026-09-20T10:00:00+00:00",
                [HumanMessage("旧问题")],
            ),
            checkpoint(
                first_thread,
                "2026-09-20T12:00:00+00:00",
                [HumanMessage("中午吃什么"), AIMessage("推荐番茄炒蛋")],
            ),
            checkpoint(
                second_thread,
                "2026-09-19T12:00:00+00:00",
                [HumanMessage("根据知识库推荐晚餐"), AIMessage("好的")],
            ),
            checkpoint("pytest-history-001", "2026-09-21T12:00:00+00:00", []),
        ]
    )

    try:
        summaries = conversation_summaries(saver, store)
    finally:
        store.close()

    assert [item["thread_id"] for item in summaries] == [first_thread, second_thread]
    assert summaries[0] == {
        "thread_id": first_thread,
        "title": "鸡蛋番茄减脂午餐",
        "updated_at": "2026-09-20T12:00:00+00:00",
        "message_count": 2,
    }
    assert summaries[1]["title"] == "根据知识库推荐晚餐"


def test_deleting_conversation_title_restores_fallback(tmp_path):
    thread_id = "33333333-3333-4333-8333-333333333333"
    store = ConversationStore(tmp_path / "conversations.sqlite")
    store.save_title(thread_id, "自定义标题")
    store.delete(thread_id)
    saver = FakeCheckpointer(
        [
            checkpoint(
                thread_id,
                "2026-09-20T12:00:00+00:00",
                [HumanMessage("清淡晚餐推荐")],
            )
        ]
    )

    try:
        summaries = conversation_summaries(saver, store)
    finally:
        store.close()

    assert summaries[0]["title"] == "清淡晚餐推荐"


def test_clean_generated_title_removes_model_formatting():
    assert clean_generated_title("标题：“西红柿鸡蛋减脂午餐。”\n这是解释") == (
        "西红柿鸡蛋减脂午餐"
    )


def test_runtime_generates_title_without_running_agent():
    class FakeModel:
        def __init__(self):
            self.messages = None

        def invoke(self, messages):
            self.messages = messages
            return AIMessage(content="标题：冰箱鸡蛋快手午餐")

    runtime = ChefRuntime.__new__(ChefRuntime)
    runtime.model = FakeModel()

    title = runtime.generate_conversation_title("冰箱有鸡蛋", "推荐西红柿炒鸡蛋")

    assert title == "冰箱鸡蛋快手午餐"
    assert len(runtime.model.messages) == 2


def test_rename_conversation_title_updates_existing_title(tmp_path):
    thread_id = "44444444-4444-4444-8444-444444444444"
    store = ConversationStore(tmp_path / "conversations.sqlite")

    class FakeAgent:
        def get_state(self, _config):
            return SimpleNamespace(values={"messages": [HumanMessage("午餐推荐")]})

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                runtime=SimpleNamespace(ready=True, agent=FakeAgent()),
                conversations=store,
            )
        )
    )

    try:
        result = rename_conversation_title(
            thread_id, ConversationTitleRequest(title="  鸡蛋减脂午餐  "), request
        )
        saved_title = store.get_title(thread_id)
    finally:
        store.close()

    assert result == {"thread_id": thread_id, "title": "鸡蛋减脂午餐"}
    assert saved_title == "鸡蛋减脂午餐"


def test_rename_conversation_title_rejects_blank_title():
    with pytest.raises(ValidationError):
        ConversationTitleRequest(title="   ")


def test_rename_conversation_title_rejects_missing_conversation(tmp_path):
    store = ConversationStore(tmp_path / "conversations.sqlite")

    class EmptyAgent:
        def get_state(self, _config):
            return SimpleNamespace(values={"messages": []})

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                runtime=SimpleNamespace(ready=True, agent=EmptyAgent()),
                conversations=store,
            )
        )
    )

    try:
        with pytest.raises(HTTPException) as exc_info:
            rename_conversation_title(
                "55555555-5555-4555-8555-555555555555",
                ConversationTitleRequest(title="新标题"),
                request,
            )
    finally:
        store.close()

    assert exc_info.value.status_code == 404
