"""Evaluate retrieval locally; optionally run paid live Agent routing checks."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.chef_core import ChefRuntime
from app.knowledge import KnowledgeError, KnowledgeService


def expected_tool_names(label: str) -> set[str]:
    return {
        "knowledge": {"search_private_knowledge"},
        "web": {"search_web_recipes"},
        "both": {"search_private_knowledge", "search_web_recipes"},
        "none": set(),
    }[label]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live-agent",
        action="store_true",
        help="调用真实模型检查工具路由与来源引用，会产生 API 用量。",
    )
    args = parser.parse_args()
    cases = json.loads((ROOT / "eval" / "rag_cases.json").read_text(encoding="utf-8"))

    knowledge = KnowledgeService.from_environment(ROOT)
    runtime = None
    try:
        if not knowledge.ready:
            print(json.dumps({"error": knowledge.initialization_error}, ensure_ascii=False, indent=2))
            return 2
        try:
            knowledge.seed_sample_documents(ROOT / "knowledge_samples")
        except KnowledgeError as exc:
            print(
                json.dumps(
                    {
                        "error": str(exc),
                        "action": "请为当前通义 API Key 授权 text-embedding-v4 后重试。",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2

        retrieval_cases = [case for case in cases if case["expected_source"]]
        retrieval_hits = 0
        details = []
        for case in retrieval_cases:
            matches = knowledge.search(case["query"], top_k=5)
            titles = {match["title"] for match in matches}
            hit = case["expected_source"] in titles
            retrieval_hits += int(hit)
            details.append({"id": case["id"], "retrieval_hit": hit, "titles": sorted(titles)})

        report = {
            "case_count": len(cases),
            "retrieval_case_count": len(retrieval_cases),
            "recall_at_5": round(retrieval_hits / max(1, len(retrieval_cases)), 4),
            "live_agent": "not_run",
            "details": details,
        }

        if args.live_agent:
            runtime = ChefRuntime(ROOT, knowledge)
            if not runtime.ready:
                report["live_agent"] = {"error": runtime.initialization_error}
            else:
                route_hits = 0
                citation_hits = 0
                live_details = []
                for case in cases:
                    result = runtime.agent.invoke(
                        {"messages": [{"role": "user", "content": case["query"]}]},
                        config={"configurable": {"thread_id": f"eval-{uuid.uuid4().hex}"}},
                    )
                    tool_names = {
                        message.name
                        for message in result["messages"]
                        if isinstance(message, ToolMessage) and message.name
                    }
                    expected = expected_tool_names(case["expected_tool"])
                    route_ok = expected.issubset(tool_names) if expected else not tool_names
                    route_hits += int(route_ok)
                    answer = str(result["messages"][-1].content)
                    citation_ok = (
                        case["expected_source"] is None
                        or case["expected_source"] in answer
                        or "第 " in answer
                    )
                    citation_hits += int(citation_ok)
                    live_details.append(
                        {"id": case["id"], "route_ok": route_ok, "tools": sorted(tool_names), "citation_ok": citation_ok}
                    )
                report["live_agent"] = {
                    "tool_selection_accuracy": round(route_hits / len(cases), 4),
                    "citation_coverage": round(citation_hits / len(cases), 4),
                    "details": live_details,
                }

        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["recall_at_5"] >= 0.8 else 1
    finally:
        if runtime:
            runtime.close()
        knowledge.close()


if __name__ == "__main__":
    raise SystemExit(main())
