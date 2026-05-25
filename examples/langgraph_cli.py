"""LangGraph Agent 命令行调试入口。

运行方式：
    python examples/langgraph_cli.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from agent_workflows.langgraph_agent import run_graph_agent
from models import llm
from vector_store import docs, splits


if __name__ == "__main__":
    print("\nLangGraph Agent CLI 初始化成功")
    print("Chat 模型：", llm.model_name)
    print("Embedding 模型：BAAI/bge-m3")
    print("原始文档数量：", len(docs))
    print("切分后的文档片段数量：", len(splits))
    print("可用工具：search_docs、calculator、get_weather")

    thread_id = "cli-thread"

    while True:
        query = input("\n请输入你的问题，输入 exit 退出：")

        if query.lower() in ["exit", "quit", "q"]:
            print("程序已退出")
            break

        if not query.strip():
            print("问题不能为空，请重新输入。")
            continue

        result = run_graph_agent(query, thread_id=thread_id)
        print("\n结构化返回结果：")
        print("answer:", result.get("answer"))
        print("tool_calls:", result.get("tool_calls"))
        print("sources:", result.get("sources"))
        print("quality_check:", result.get("quality_check"))
