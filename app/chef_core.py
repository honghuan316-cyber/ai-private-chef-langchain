"""LangChain Agent runtime and retrieval tools for AI Private Chef."""

from __future__ import annotations

import json
import os
import sys
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver

from app.knowledge import KnowledgeService


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SYSTEM_PROMPT = """
你是“AI 私厨”，一名严谨、友好、实用的中文私人厨师。

你可以使用两个不同的信息来源：
1. search_private_knowledge：检索用户上传的本地菜谱、营养、忌口和食品安全资料。
2. search_web_recipes：使用 Tavily 搜索互联网中的菜谱和外部参考资料。

工具选择规则：
- 用户询问营养搭配、减脂饮食、忌口、食物过敏、食品安全或用户资料中的内容时，必须先检索本地知识库，再依据检索结果回答。
- 用户说“根据我的资料”“知识库里怎么说”“文档中有没有”时，也必须检索本地知识库。
- 用户要求最新菜谱、网络做法或外部链接时，使用网络搜索。
- 用户要求依据个人营养资料推荐网络菜谱时，先检索知识库，再搜索网络资料。
- 用户追问“刚才第二道菜”等已有内容时，优先利用当前 thread_id 的会话历史，不要重复搜索。
- 普通问候不调用工具。
- 工具没有找到可靠资料时要明确说明，绝不能编造文档、页码或网址。
- 只要使用了本地知识库，最终回答必须至少标注一次完整的“[文档名，第 X 页]”；只要使用了网络搜索，最终回答必须保留真实可点击链接。

工作流程：
1. 如果用户提供图片，先识别图片中清晰可见的食材；不确定的食材必须标注“不确定”。
2. 需要推荐菜谱时，综合食材匹配度 50%、营养均衡 30%、制作难度 20% 进行评价和排序。
3. 本地知识库来源使用“[文档名，第 X 页]”标注；网络来源使用可点击的 Markdown 链接。
4. 不输出工具调用过程、内部推理或原始工具 JSON。

首次推荐菜谱时使用以下结构：
## 食材识别
- 已识别食材：...
- 需求与限制：...

## 推荐结果
### 1. 菜名
- **推荐理由**：...
- **综合评分**：X/10（匹配度 X/10｜营养 X/10｜难度 X/10）
- **制作难度**：简单/中等/较难
- **预计用时**：约 X 分钟
- **主要食材**：...
- **制作步骤**：使用 3—6 个清晰的编号步骤
- **营养评价**：说明蛋白质、蔬菜、油盐等情况，不虚构精确克数或医疗结论
- **参考来源**：真实的本地文档引用或网络链接

最后增加“## 私厨建议”，给出替换食材、忌口或减脂方面的一条实用建议。
默认推荐 2 道菜；用户指定数量时按用户要求。全程使用简体中文。
""".strip()


KNOWLEDGE_ROUTE_KEYWORDS = (
    "营养",
    "正餐",
    "食物类别",
    "减脂",
    "蛋白",
    "主食",
    "盐和油",
    "控盐",
    "控油",
    "过敏",
    "忌口",
    "食品安全",
    "生肉",
    "生食",
    "剩菜",
    "熟没熟",
    "交叉污染",
    "交叉接触",
    "根据我的资料",
    "根据我的营养资料",
    "知识库",
    "文档中",
)
WEB_ROUTE_KEYWORDS = ("最新", "网上", "网络", "外部链接", "搜索", "链接")


class PrivateChefRoutingMiddleware(AgentMiddleware):
    """Force evidence lookup for safety-critical domains, while preserving Agent autonomy."""

    name = "private_chef_routing"

    @staticmethod
    def _tool_name(candidate: Any) -> str | None:
        if isinstance(candidate, dict):
            function = candidate.get("function")
            if isinstance(function, dict):
                return function.get("name")
            return candidate.get("name")
        return getattr(candidate, "name", None)

    @staticmethod
    def _routed_request(request: ModelRequest) -> ModelRequest:
        latest_user_index = -1
        latest_query = ""
        for index, message in enumerate(request.messages):
            if isinstance(message, HumanMessage):
                latest_user_index = index
                latest_query = message.text

        if latest_user_index < 0:
            return request

        messages_after_user = request.messages[latest_user_index + 1 :]
        used_tools = {
            message.name
            for message in messages_after_user
            if isinstance(message, ToolMessage) and message.name
        }
        needs_knowledge = any(keyword in latest_query for keyword in KNOWLEDGE_ROUTE_KEYWORDS)
        needs_web = any(keyword in latest_query for keyword in WEB_ROUTE_KEYWORDS)

        target_tool = None
        if needs_knowledge and "search_private_knowledge" not in used_tools:
            target_tool = "search_private_knowledge"
        elif needs_web and "search_web_recipes" not in used_tools:
            target_tool = "search_web_recipes"

        if not target_tool:
            return request

        selected_tools = [
            candidate
            for candidate in request.tools
            if PrivateChefRoutingMiddleware._tool_name(candidate) == target_tool
        ]
        if not selected_tools:
            return request
        # DashScope's OpenAI-compatible endpoint consistently honors `required`
        # when the available tool list contains only the policy-selected tool.
        return request.override(tools=selected_tools, tool_choice="required")

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._routed_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._routed_request(request))


class PolicyRoutedAgent:
    """Add deterministic evidence calls before the model for critical query types."""

    def __init__(self, graph, tools: list[Any]) -> None:
        self.graph = graph
        self.tools = {item.name: item for item in tools}

    @staticmethod
    def _query_from_input(agent_input: dict[str, Any]) -> str:
        for message in reversed(agent_input.get("messages", [])):
            if isinstance(message, HumanMessage):
                return message.text
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    return content
        return ""

    def _required_tools(self, query: str) -> list[str]:
        required = []
        if any(keyword in query for keyword in KNOWLEDGE_ROUTE_KEYWORDS):
            required.append("search_private_knowledge")
        if any(keyword in query for keyword in WEB_ROUTE_KEYWORDS):
            required.append("search_web_recipes")
        return [name for name in required if name in self.tools]

    def _prepare(self, agent_input: dict[str, Any]) -> tuple[dict[str, Any], list[ToolMessage]]:
        query = self._query_from_input(agent_input)
        required = self._required_tools(query)
        if not required:
            return agent_input, []

        calls = []
        tool_messages = []
        for name in required:
            call_id = f"policy-{uuid.uuid4().hex}"
            tool_call = {
                "name": name,
                "args": {"query": query},
                "id": call_id,
                "type": "tool_call",
            }
            calls.append(tool_call)
            try:
                result = self.tools[name].invoke(tool_call)
            except Exception:
                result = ToolMessage(
                    content="资料检索暂时失败，请明确说明当前无法取得该来源。",
                    tool_call_id=call_id,
                    name=name,
                    artifact={"sources": []},
                )
            tool_messages.append(result)

        prepared = dict(agent_input)
        prepared["messages"] = [
            *agent_input.get("messages", []),
            AIMessage(content="", tool_calls=calls),
            *tool_messages,
        ]
        return prepared, tool_messages

    def invoke(self, agent_input: dict[str, Any], *args, **kwargs):
        prepared, _ = self._prepare(agent_input)
        return self.graph.invoke(prepared, *args, **kwargs)

    def stream(self, agent_input: dict[str, Any], *args, **kwargs):
        prepared, tool_messages = self._prepare(agent_input)
        if kwargs.get("stream_mode") == "messages":
            for message in tool_messages:
                yield message, {"langgraph_node": "policy_router"}
        yield from self.graph.stream(prepared, *args, **kwargs)

    def get_state(self, *args, **kwargs):
        return self.graph.get_state(*args, **kwargs)


def create_knowledge_tool(knowledge: KnowledgeService):
    @tool(
        "search_private_knowledge",
        response_format="content_and_artifact",
        description=(
            "检索用户私人烹饪知识库。适合回答用户上传的菜谱、营养、忌口、"
            "食品安全资料中的内容。输入应是完整、明确的中文检索问题。"
        ),
    )
    def search_private_knowledge(query: str) -> tuple[str, dict[str, Any]]:
        matches = knowledge.search(query)
        if not matches:
            return (
                "知识库没有找到相关资料。请不要编造本地文档来源或页码。",
                {"sources": []},
            )
        sections = []
        sources = []
        for item in matches:
            sections.append(
                f"来源：{item['title']}，第 {item['page']} 页，相关度 {item['score']:.2f}\n"
                f"内容：{item['content']}"
            )
            sources.append(
                {
                    "source_type": "knowledge",
                    "title": item["title"],
                    "document_id": item["document_id"],
                    "page": item["page"],
                    "score": item["score"],
                    "url": item["url"],
                }
            )
        return "\n\n---\n\n".join(sections), {"sources": sources}

    return search_private_knowledge


def _parse_tavily_results(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, dict):
        values = raw.get("results", [])
    elif isinstance(raw, list):
        values = raw
    else:
        values = []
    return [item for item in values if isinstance(item, dict)]


def create_web_search_tool():
    tavily = TavilySearch(max_results=5, topic="general")

    @tool(
        "search_web_recipes",
        response_format="content_and_artifact",
        description=(
            "使用 Tavily 搜索互联网中的真实菜谱、做法和外部参考链接。"
            "适合最新信息、网络菜谱和需要可点击来源的问题。"
        ),
    )
    def search_web_recipes(query: str) -> tuple[str, dict[str, Any]]:
        raw = tavily.invoke({"query": query})
        results = _parse_tavily_results(raw)
        if not results:
            return "网络搜索没有找到可靠结果，请不要编造网址。", {"sources": []}
        sections = []
        sources = []
        for item in results[:5]:
            title = str(item.get("title") or "网络资料")
            url = str(item.get("url") or "")
            content = str(item.get("content") or item.get("snippet") or "").strip()
            sections.append(f"标题：{title}\n链接：{url}\n摘要：{content}")
            if url:
                sources.append(
                    {
                        "source_type": "web",
                        "title": title,
                        "document_id": None,
                        "page": None,
                        "score": item.get("score"),
                        "url": url,
                    }
                )
        return "\n\n---\n\n".join(sections), {"sources": sources}

    return search_web_recipes


class ChefRuntime:
    """Lifecycle-owned model, tools, agent, and SQLite checkpointer."""

    def __init__(self, root: Path, knowledge: KnowledgeService) -> None:
        self.root = Path(root)
        self.knowledge = knowledge
        self.agent = None
        self.checkpointer = None
        self.initialization_error: str | None = None
        self._checkpointer_context = None

        api_key = os.getenv("QWEN_API_KEY")
        if not api_key:
            self.initialization_error = "未配置通义千问 API Key。"
            return

        try:
            model = ChatOpenAI(
                model=os.getenv("QWEN_MODEL", "qwen-plus"),
                api_key=api_key,
                base_url=os.getenv(
                    "QWEN_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
                temperature=0.25,
                timeout=90,
                max_retries=2,
                streaming=True,
            )
            tools = []
            if knowledge.ready:
                tools.append(create_knowledge_tool(knowledge))
            if os.getenv("TAVILY_API_KEY"):
                tools.append(create_web_search_tool())

            database_path = self.root / "data" / "chef_memory.sqlite"
            database_path.parent.mkdir(exist_ok=True)
            self._checkpointer_context = SqliteSaver.from_conn_string(str(database_path))
            self.checkpointer = self._checkpointer_context.__enter__()
            graph = create_agent(
                model=model,
                tools=tools,
                system_prompt=SYSTEM_PROMPT,
                middleware=[PrivateChefRoutingMiddleware()],
                checkpointer=self.checkpointer,
            )
            self.agent = PolicyRoutedAgent(graph, tools)
        except Exception:
            self.initialization_error = "AI Agent 初始化失败，请检查模型与数据库配置。"
            self.close()

    @property
    def ready(self) -> bool:
        return self.agent is not None

    def close(self) -> None:
        if self._checkpointer_context is not None:
            try:
                self._checkpointer_context.__exit__(None, None, None)
            finally:
                self._checkpointer_context = None
                self.checkpointer = None
