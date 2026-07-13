import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SYSTEM_PROMPT = """
你是“AI 私厨”，一名严谨、友好、实用的中文私人厨师。

工作流程：
1. 如果用户提供图片，先识别图片中清晰可见的食材；不确定的食材必须说明“不确定”，不能臆测。
2. 当用户要求推荐菜谱、做法或食材搭配时，必须优先调用 Tavily 搜索真实资料。
3. 综合搜索结果，按照食材匹配度 50%、营养均衡 30%、制作难度 20%进行评价和排序。
4. 利用当前会话历史回答“刚才第二道菜”等连续追问。
5. 不输出工具调用过程、内部推理或原始搜索 JSON。

首次推荐菜谱时，使用以下中文 Markdown 结构：
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
- **营养评价**：说明蛋白质、蔬菜、油盐等情况；不要虚构精确克数或医疗结论
- **参考来源**：[页面名称](真实 URL)

最后增加“## 私厨建议”，给出替换食材、忌口或减脂方面的一条实用建议。
引用必须来自搜索结果中的真实链接。若搜索失败，要明确说明无法取得外部来源，不能编造网址。
默认推荐 2 道菜；用户指定数量时按用户要求。全程使用简体中文。
""".strip()

model = ChatOpenAI(
    model=os.getenv("QWEN_MODEL", "qwen-plus"),
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    temperature=0.25,
    timeout=90,
    max_retries=2,
    streaming=True,
)

search_tool = TavilySearch(max_results=5, topic="general")
_checkpointer_context = None


def get_agent():
    global _checkpointer_context
    if _checkpointer_context is None:
        database_path = ROOT / "data" / "chef_memory.sqlite"
        database_path.parent.mkdir(exist_ok=True)
        _checkpointer_context = SqliteSaver.from_conn_string(str(database_path))
    checkpointer = _checkpointer_context.__enter__()
    agent = create_agent(
        model=model,
        tools=[search_tool],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent, checkpointer
