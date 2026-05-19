"""手写 Function Calling Agent Loop。

这个文件是当前项目的核心：
- 模型负责“决定是否调用工具、调用什么工具、传什么参数”
- 程序负责“真正执行 Python 工具函数”
- 工具结果会被写回 messages，再交给模型生成最终回答

当前能力：
- 支持 search_docs / calculator / get_weather 三个工具
- 支持多轮工具调用
- 支持工具调用记录
- 支持 RAG 来源追踪
- 支持工具异常保护
- 支持 JSONL 日志写入
"""

import json
import os

from config import MAX_AGENT_ROUNDS
from models import llm
from schemas import TOOLS
from tools import AVAILABLE_FUNCTIONS


def save_agent_log(record: dict) -> None:
    """把 Agent 运行结果追加写入 logs/agent.log。

    日志格式是 JSONL：
    - 一行是一条 JSON 记录
    - 便于追加写入
    - 便于后续用脚本做分析和评估
    """
    # 确保 logs 文件夹存在；如果已存在，不会报错。
    os.makedirs("logs", exist_ok=True)

    # ensure_ascii=False 可以保证中文正常写入，而不是变成 Unicode 转义。
    with open("logs/agent.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_agent(user_query: str) -> dict:
    """
    运行一轮 Agent 问答。

    Agent Loop 流程：
    1. 把用户问题发给模型
    2. 模型判断是否需要调用工具
    3. 如果需要，程序执行工具，并把工具结果放回 messages
    4. 再次调用模型
    5. 重复以上过程，直到模型不再调用工具，输出最终回答

    返回结构：
    {
        "user_query": "用户原始问题",
        "answer": "模型最终回答",
        "tool_calls": [
            {
                "name": "工具名",
                "args": "工具参数",
                "result": "工具执行结果"
            }
        ],
        "sources": ["RAG 检索来源"],
        "rounds": 2
    }
    """
    # 记录每次工具调用的工具名、参数和结果。
    # 这个列表最终会返回给 API，也会写入日志，方便调试和评估。
    tool_call_records = []

    # 记录 search_docs 检索到的文档来源，方便 API 返回和答案溯源。
    sources = []

    # messages 是 Agent 的上下文记忆。
    # 每一轮都会追加：
    # - 模型消息 AIMessage
    # - 工具结果 tool message
    # 这样模型下一轮才能知道之前发生了什么。
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
                "如果使用了 search_docs 的结果，请在回答末尾列出参考来源。"
                "如果知识库中没有相关信息，请明确说明：知识库中没有找到相关信息，不要编造。"
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

            # 先构造统一结果对象，再写日志、再返回。
            # 这样可以保证 API 返回和日志记录使用同一份数据。
            result = {
                "user_query": user_query,
                "answer": message.content,
                "tool_calls": tool_call_records,
                "sources": sources,
                "rounds": round_num,
            }

            save_agent_log(result)
            return result

        print(f"\n模型决定调用 {len(message.tool_calls)} 个工具")

        # 这一步很重要：
        # 必须先把模型的 tool_calls 请求加入 messages，
        # 后面再追加 tool 结果，模型才能正确关联“请求”和“结果”。
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
            # 注意：工具名来自模型输出，不能假设一定正确。
            if function_name not in AVAILABLE_FUNCTIONS:
                function_response = {"error": f"未知工具：{function_name}"}
            else:
                # 根据工具名找到真正的 Python 函数，并把参数展开传进去。
                # 例如：calculator(**{"operation": "add", "a": 1, "b": 2})
                # function_response = AVAILABLE_FUNCTIONS[function_name](**function_args)
                try:
                    function_response = AVAILABLE_FUNCTIONS[function_name](**function_args)
                except Exception as e:
                    function_response = {"error": f"工具执行出错：{str(e)}"}

            print("工具执行结果：", function_response)

            # 记录工具调用详情。
            # 这相当于一个最小 trace，可以用于：
            # - API 返回
            # - 日志记录
            # - 自动评估
            # - 排查模型是否选错工具或传错参数
            tool_call_records.append(
                {
                    "name": function_name,
                    "args": function_args,
                    "result": function_response,
                }
            )

            # 如果是 search_docs 工具，就从检索结果里提取 source。
            # function_response 正常情况下是 list[dict]；
            # 如果工具报错，可能是 {"error": "..."}，所以要先判断类型。
            if function_name == "search_docs" and isinstance(function_response, list):
                for item in function_response:
                    source = item.get("source")
                    if source and source not in sources:
                        sources.append(source)

            # 把工具执行结果返回给模型。
            # 这里的 role 必须是 tool，并且必须带 tool_call_id。
            # tool_call_id 用来告诉模型：这个结果对应前面哪一次工具调用。
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(function_response, ensure_ascii=False),
                }
            )

    print("\n达到最大 Agent 轮数，程序停止继续调用工具。")

    # 即使达到最大轮数，也返回和正常情况一致的字段，
    # 这样 API 调用方不用写额外的特殊处理逻辑。
    result = {
        "user_query": user_query,
        "answer": "达到最大 Agent 轮数，程序停止继续调用工具。",
        "tool_calls": tool_call_records,
        "sources": sources,
        "rounds": MAX_AGENT_ROUNDS,
    }

    save_agent_log(result)
    return result
