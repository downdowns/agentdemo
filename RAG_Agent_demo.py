"""RAG Agent Demo 主入口。"""

from agent import run_agent
from models import llm
from vector_store import docs, splits


if __name__ == "__main__":
    print("\nRAG Agent Demo 初始化成功")
    print("Chat 模型：", llm.model_name)
    print("Embedding 模型：BAAI/bge-m3")
    print("Embedding 类型：HuggingFaceEmbeddings")
    print("原始文档数量：", len(docs))
    print("切分后的文档片段数量：", len(splits))
    print("可用工具：search_docs、calculator、get_weather")

    # 打印切分后的文档片段，方便你观察知识库到底被切成了什么样。
    for i, split in enumerate(splits):
        print(f"\n--- 文档片段 {i + 1} ---")
        print("来源：", split.metadata)
        print("内容：", split.page_content)

    print("\nChroma 向量数据库准备完成")

    # 进入命令行交互循环：用户可以一直提问。
    while True:
        # 接收用户输入。
        query = input("\n请输入你的问题，输入 exit 退出：")

        # 用户输入 exit / quit / q 时退出程序。
        if query.lower() in ["exit", "quit", "q"]:
            print("程序已退出")
            break

        # 如果用户只按回车，没有输入实际问题，就跳过本轮。
        if not query.strip():
            print("问题不能为空，请重新输入。")
            continue

        # 把用户问题交给 Agent 处理。
        result = run_agent(query)

        print("\n 结构化返回结果：")
        print("answer:", result["answer"])
        print("tool_calls:", result["tool_calls"])
        print("sources:", result["sources"])
        print("rounds:", result["rounds"])
