"""LangGraph RAG Agent Demo 主入口。

运行方式：
    python LangGraph_RAG_Agent_demo.py
"""

from langgraph_agent import run_langgraph_agent
from models import llm
from vector_store import docs, splits


if __name__ == "__main__":
    print("\nLangGraph RAG Agent Demo 初始化成功")
    print("Chat 模型：", llm.model_name)
    print("Embedding 模型：BAAI/bge-m3")
    print("原始文档数量：", len(docs))
    print("切分后的文档片段数量：", len(splits))
    print("可用工具：search_docs、calculator、get_weather")

    while True:
        query = input("\n请输入你的问题，输入 exit 退出：")

        if query.lower() in ["exit", "quit", "q"]:
            print("程序已退出")
            break

        if not query.strip():
            print("问题不能为空，请重新输入。")
            continue

        run_langgraph_agent(query)
