"""Agent Loop：模型决定是否调用工具，程序负责执行工具。"""

import json

from config import MAX_AGENT_ROUNDS
from models import llm
from schemas import TOOLS
from tools import AVAILABLE_FUNCTIONS


def run_agent(user_query: str) -> None:
    """
    运行一轮 Agent 问答。

    Agent Loop 流程：
    1. 把用户问题发给模型
    2. 模型判断是否需要调用工具
    3. 如果需要，程序执行工具，并把工具结果放回 messages
    4. 再次调用模型
    5. 重复以上过程，直到模型不再调用工具，输出最终回答
    """
    # messages 是对话历史。
    # Agent 每一轮都会把模型回复、工具结果追加进去，形成上下文记忆。
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个多工具 Agent 助手。"
                "如果用户的问题包含多个彼此独立的任务，请尽量在同一轮中一次性调用所有需要的工具。"
                "如果问题需要查本地知识库，请调用 search_docs。"
                "如果问题需要数学计算，请调用 calculator。"
                "如果问题需要查询天气，请调用 get_weather。"
                "只有当前一个工具结果会影响下一个工具参数时，才分多轮调用工具。"
                "请根据工具结果给出清晰、简洁的最终回答。"
            ),
        },
        {
            "role": "user",
            "content": user_query,
        },
    ]

    # 最多循环 MAX_AGENT_ROUNDS 轮，防止模型无限调用工具。
    for round_num in range(1, MAX_AGENT_ROUNDS + 1):
        print(f"\n========== Agent 第 {round_num} 轮 ==========")

        # 调用模型，并把工具 schema 传给模型。
        # tool_choice="auto" 表示让模型自己决定是否调用工具。
        response = llm.invoke(
            messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        # LangChain 返回的是 AIMessage，这里命名为 message，方便理解。
        message = response

        # 如果模型没有继续调用工具，说明它已经准备好最终回答。
        if not message.tool_calls:
            print("\n模型最终回答：")
            print(message.content)
            return

        print(f"\n模型决定调用 {len(message.tool_calls)} 个工具")

        # 这一步很重要：把模型的 tool_calls 请求加入对话历史。
        messages.append(message)

        # 模型可能一次请求调用多个工具，所以这里逐个执行。
        for tool_call in message.tool_calls:
            # 工具名称，例如 search_docs / calculator / get_weather。
            function_name = tool_call["name"]

            # 工具参数，例如 {"city": "上海"}。
            function_args = tool_call["args"]

            # 工具调用 ID：后面把工具结果返回给模型时必须带上，
            # 模型用它来对应“哪个工具调用”得到了“哪个工具结果”。
            tool_call_id = tool_call["id"]

            print("\n--- 准备执行工具 ---")
            print("工具名称：", function_name)
            print("工具参数：", function_args)

            # 防御性判断：如果模型返回了不存在的工具名，给它一个错误结果。
            if function_name not in AVAILABLE_FUNCTIONS:
                function_response = {"error": f"未知工具：{function_name}"}
            else:
                # 根据工具名找到真正的 Python 函数，并把参数展开传进去。
                # 例如：calculator(**{"operation": "add", "a": 1, "b": 2})
                function_response = AVAILABLE_FUNCTIONS[function_name](**function_args)

            print("工具执行结果：", function_response)

            # 把工具执行结果返回给模型。
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(function_response, ensure_ascii=False),
                }
            )

    print("\n达到最大 Agent 轮数，程序停止继续调用工具。")
