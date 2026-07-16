# RAG 评测说明

默认评测只调用 Embedding 检索，统计 15 条用例中的 Recall@5：

```powershell
python eval/evaluate_rag.py
```

如果需要同时评估 Agent 是否正确选择知识库、Tavily 或不调用工具，可执行：

```powershell
python eval/evaluate_rag.py --live-agent
```

`--live-agent` 会调用真实通义千问与 Tavily，产生 API 用量，因此不放进日常 Pytest。

验收目标：

- Recall@5 不低于 0.80。
- 需要本地资料的回答能够标注文档名和页码。
- 最新网络资料使用 Tavily，不伪造链接。
- 普通问候不调用检索工具。

