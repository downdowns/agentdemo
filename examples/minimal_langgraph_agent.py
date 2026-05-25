"""最小 LangGraph Agent 示例，保留为 examples 参考。

这个文件用 LangGraph 的 StateGraph 替代原来的手写 Agent Loop。
核心思想：
- State：保存对话消息 messages
- Node：模型节点 / 工具节点
- Edge：节点之间的流转关系
- Conditional Edge：根据模型是否返回 tool_calls 决定下一步
- Checkpoint：保存每一步图状态，支持有状态 Agent
"""

import json
from typing import Literal

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models import llm
from schemas import TOOLS
from tools import AVAILABLE_FUNCTIONS


SYSTEM_PROMPT = (
    "你是一个多工具 RAG Agent 助手。"
    "如果用户的问题需要查询本地知识库，请调用 search_docs。"
    "如果用户的问题需要数学计算，请调用 calculator。"
    "如果用户的问题需要查询天气，请调用 get_weather。"
    "如果用户的问题包含多个彼此独立的任务，请尽量在同一轮中一次性调用所有需要的工具。"
    "只有当前一个工具结果会影响下一个工具参数时，才分多轮调用工具。"
    "请根据工具结果给出清晰、简洁的最终回答。"
)


def call_model(state: MessagesState) -> dict:
    """模型节点：根据当前状态决定是否回答或调用工具。

    参数：
    - state：LangGraph 当前状态，里面最重要的是 messages。

    返回：
    - {"messages": [response]}：LangGraph 会把新的 AIMessage 追加到状态里。
    """
    # 从状态里取出历史消息。
    messages = state["messages"]

    # 每次调用模型时，都把 system prompt 放在最前面。
    # 这样模型知道自己有哪些工具、应该如何使用工具。
    model_input = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    # 把工具 schema 传给模型，让模型自己决定是否调用工具。
    response = llm.invoke(
        model_input,
        tools=TOOLS,
        tool_choice="auto",
    )

    return {"messages": [response]}


def call_tools(state: MessagesState) -> dict:
    """工具节点：执行模型上一条消息里请求调用的所有工具。

    LangGraph 中模型节点只负责产生 tool_calls，
    真正执行工具的逻辑放在这个 tools 节点里。
    """
    # 取出最新的一条模型消息。
    last_message = state["messages"][-1]

    tool_messages = []

    # 模型可能一次请求多个工具，所以逐个执行。
    for tool_call in last_message.tool_calls:
        function_name = tool_call["name"]
        function_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        print("\n--- LangGraph 准备执行工具 ---")
        print("工具名称：", function_name)
        print("工具参数：", function_args)

        # 防御性判断：如果模型返回了不存在的工具名，返回结构化错误。
        if function_name not in AVAILABLE_FUNCTIONS:
            function_response = {"error": f"未知工具：{function_name}"}
        else:
            try:
                # 根据工具名找到真实 Python 函数，并把参数展开传进去。
                function_response = AVAILABLE_FUNCTIONS[function_name](**function_args)
            except Exception as e:
                # 工具异常不能让整个图崩溃，而是作为工具结果返回给模型。
                function_response = {"error": str(e)}

        print("工具执行结果：", function_response)

        # ToolMessage 必须带 tool_call_id，用来和模型的某个 tool_call 对应。
        tool_messages.append(
            ToolMessage(
                tool_call_id=tool_call_id,
                content=json.dumps(function_response, ensure_ascii=False),
            )
        )

    return {"messages": tool_messages}


def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    """条件边：根据最新模型消息决定下一步走向。

    - 如果模型返回 tool_calls：继续去 tools 节点执行工具。
    - 如果模型没有返回 tool_calls：说明已经可以最终回答，结束图。
    """
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return END


def build_graph():
    """构建并编译 LangGraph Agent。"""
    # StateGraph(MessagesState) 表示图的状态主要由 messages 组成。
    workflow = StateGraph(MessagesState)

    # 添加两个节点：模型节点和工具节点。
    workflow.add_node("model", call_model)
    workflow.add_node("tools", call_tools)

    # 图从 START 进入 model 节点。
    workflow.add_edge(START, "model")

    # model 节点之后走条件边：
    # - 有 tool_calls：去 tools
    # - 没有 tool_calls：结束
    workflow.add_conditional_edges(
        "model",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )

    # 工具执行完后，再回到模型节点，让模型基于工具结果继续判断或最终回答。
    workflow.add_edge("tools", "model")

    # Checkpoint：保存每一步状态。
    # 这里用内存版 InMemorySaver，适合 demo。
    # 生产中可以换成 SQLite / Postgres 等持久化 checkpointer。
    checkpointer = InMemorySaver()

    return workflow.compile(checkpointer=checkpointer)


# 编译好的 LangGraph app。
graph_app = build_graph()


def run_langgraph_agent(user_query: str, thread_id: str = "demo-thread") -> None:
    """运行 LangGraph Agent。

    thread_id 用于 checkpoint 区分不同会话。
    同一个 thread_id 可以保留同一条对话线程的状态。
    """
    # LangGraph 的输入状态。这里传入用户消息。
    inputs = {
        "messages": [
            {
                "role": "user",
                "content": user_query,
            }
        ]
    }

    # configurable.thread_id 是 checkpointer 需要的线程 ID。
    # recursion_limit 用来防止图无限循环。
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 10,
    }

    # invoke 会运行整张图，直到走到 END。
    final_state = graph_app.invoke(inputs, config=config)

    # 最后一条消息通常就是模型最终回答。
    final_message = final_state["messages"][-1]

    print("\nLangGraph 模型最终回答：")
    print(final_message.content)
