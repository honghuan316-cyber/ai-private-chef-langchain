from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from app.chef_core import ChefRuntime
from app.conversations import (
    ConversationStore,
    clean_generated_title,
    conversation_summaries,
)


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
