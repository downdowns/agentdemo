"""
LangGraph Step 2:把手写 Agent Loop 改造成图结构

目标：
- 理解 State/Node/Edge/Conditional Edge
- 实现 model -> tools -> model 的最小Agent图
"""
import json
import sys
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models import llm
from schemas import TOOLS
from tools import AVAILABLE_FUNCTIONS

SYSTEM_PROMPT = (
    "你是一个多工具 Agent 助手。"
    "如果问题需要查本地知识库，请调用 search_docs。"
    "如果问题需要数学计算，请调用 calculator。"
    "如果问题需要查询天气，请调用 get_weather。"
    "请根据工具结果给出清晰、简洁的最终回答。"
)

def merge_tool_calls(old: list[dict] | None, new: list[dict] | None) -> list[dict]:
    """合并工具调用记录，让 tool_calls 可以跨节点、跨轮次累积。"""
    return (old or []) + (new or [])


def merge_sources(old: list[str] | None, new: list[str] | None) -> list[str]:
    """合并 RAG 来源，并去重。"""
    merged = []

    for source in (old or []) + (new or []):
        if source and source not in merged:
            merged.append(source)

    return merged

class GraphState(TypedDict):
    """LangGraph 自定义状态。

    messages：
        对话消息列表。Annotated + add_messages 表示：
        节点返回新的 messages 时，不是覆盖旧消息，而是追加合并。

    tool_calls：
        结构化工具调用记录。

    sources：
        RAG 检索来源。
    """

    messages: Annotated[list, add_messages]
    tool_calls: Annotated[list[dict], merge_tool_calls]
    sources: Annotated[list[str], merge_sources]

def call_model(state: GraphState) -> dict:
    """
    模型节点：根据当前 messages 判断是回答还是调用工具。

    输入：
    - state["messages"]：当前图状态里的对话历史

    输出：
    - {"messages": [response]}：把模型新消息追加到状态中
    """

    # 取出当前对话历史。
    messages = state["messages"]

    # 每次调用模型前，把 system prompt 放在最前面。
    # 区别是： LangGraph 的 messages state 里通常只放用户、模型、工具消息；
    # system prompt 可以在每次调用模型时临时拼上去。
    model_input = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm.invoke(
        model_input,
        tools=TOOLS,
        tool_choice="auto",
    )

    # 意思是：把模型返回的新消息追加到state["messages"]中
    return {"messages": [response]}

def call_tools(state: GraphState) -> dict:
    """
    工具节点： 执行模型上一条消息请求调用的工具

    输入： - state["messages"][-1]: 最新模型消息，里面可能包含 tool_calls

    输出：- {"messages": tool_messages}：把工具结果追加到状态中
    """
    last_message = state["messages"][-1]

    tool_messages = []
    tool_call_records = []
    new_sources = []

    for tool_call in last_message.tool_calls:
        function_name = tool_call["name"]
        function_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        print("\n--- LangGraph 准备执行工具 ------")
        print("工具名称：", function_name)
        print("工具参数：", function_args)

        if function_name not in AVAILABLE_FUNCTIONS:
            function_response = {"error": f"未知工具：{function_name}"}
        else:
            try:
                function_response = AVAILABLE_FUNCTIONS[function_name](**function_args)
            except Exception as e:
                function_response = {"error": f"工具执行出错：{str(e)}"}
        
        print("工具执行结果：", function_response)

        tool_call_records.append(
            {
                "name": function_name,
                "args": function_args,
                "result": function_response,
            }
        )
        
        if function_name == "search_docs" and isinstance(function_response, list):
            for item in function_response:
                source = item.get("source")
                if source and source not in new_sources:
                    new_sources.append(source)

        tool_messages.append(
            ToolMessage(
                tool_call_id = tool_call_id,
                content = json.dumps(function_response,ensure_ascii=False),
            )
        )
    
    return {"messages": tool_messages,
            "tool_calls": tool_call_records,
            "sources": new_sources,
            }

def should_continue(state: GraphState) -> Literal["tools", "__end__"]:
    """
    条件边：判断模型是否还要调用工具

    返回：
    - "tools": 如果最新模型消息里有tool_calls，下一步去工具节点
    - END：如果没有 tool_calls，说明模型已经给出最终回答，结束图
    """
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"
    
    return END

def build_graph():
    """
    构建并编译 LangGraph Agent 图。

    图结构：
    START → model → 条件判断
                    ├── tools → model
                    └── END
    """
    workflow = StateGraph(GraphState)
    # 添加节点。
    workflow.add_node("models", call_model)
    workflow.add_node("tools", call_tools)

    # 从 START 进入 model节点
    workflow.add_edge(START, "models")

    # model 节点执行完后，根据 should_continue 的返回值选择下一步
    workflow.add_conditional_edges(
        "models",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )

    # tools 执行完后，回到model，让模型基于工具结果继续判断或回答
    workflow.add_edge("tools", "models")
    
    # InMemorySaver 表示状态保存在当前 Python 进程内存里
    # 特点：适合学习和demo，程序一关，状态消失
    # 严格来说，这个“退出再启动”的测试，不能验证跨进程持久化
    checkpointer = InMemorySaver()

    return workflow.compile(checkpointer=checkpointer)

graph_app = build_graph()

# 添加了 thread_id ，作用是区分不同会话
def run_graph_agent(user_query: str, thread_id: str = "demo-thread") -> dict:
    """运行 LangGraph Agent。"""
    inputs = {
        "messages": [
            {
                "role": "user",
                "content": user_query,
            }
        ],
        "tool_calls": [],
        "sources": [],
    }

    config = {
        "configurable":{
            "thread_id":thread_id,
        }
    }

    final_state = graph_app.invoke(inputs, config=config)

    final_message = final_state["messages"][-1]

    # messages_count：当前会话累计消息数量
    # 这个字段可以帮助我观察 checkpoint 是否在累积历史
    result = {
        "thread_id": thread_id,
        "answer": final_message.content,
        "tool_calls": final_state.get("tool_calls", []),
        "sources": final_state.get("sources", []),
        "messages_count": len(final_state["messages"]),
    }

    print("\nLangGraph 最终回答：")
    print(result["answer"])
    print("工具调用：", result["tool_calls"])
    print("来源：", result["sources"])
    print("messages_count：", result["messages_count"])

    return result

def test_thread_isolation() -> None:
    """测试不同 thread_id 之间的会话隔离。"""
    print("\n========== 测试 thread_id 会话隔离 ==========")

    print("\n[user-a] 第一轮：告诉 Agent 名字")
    result_a1 = run_graph_agent("我叫小明", thread_id="user-a")
    print("结构化返回：", result_a1)

    print("\n[user-b] 第一轮：询问名字")
    result_b1 = run_graph_agent("我叫什么？", thread_id="user-b")
    print("结构化返回：", result_b1)

    print("\n[user-a] 第二轮：询问名字")
    result_a2 = run_graph_agent("我叫什么？", thread_id="user-a")
    print("结构化返回：", result_a2)

def test_tool_state_accumulation() -> None:
    """测试 tool_calls 和 sources 是否会在同一个 thread_id 下累积。"""
    print("\n========== 测试 tool_calls / sources 状态累积 ==========")

    thread_id = "tool-state-test"

    print("\n第一轮：计算问题")
    result_1 = run_graph_agent("帮我计算 23 乘以 19", thread_id=thread_id)
    print("第一轮 tool_calls:", result_1["tool_calls"])
    print("第一轮 sources:", result_1["sources"])

    print("\n第二轮：RAG 问题")
    result_2 = run_graph_agent("RAG 是什么？", thread_id=thread_id)
    print("第二轮 tool_calls:", result_2["tool_calls"])
    print("第二轮 sources:", result_2["sources"])


def main_chat_loop() -> None:
    """命令行交互模式。

    这个函数只负责正常聊天，不自动运行测试。

    如果要测试 checkpoint / thread_id，可以临时在 __main__ 中改成：
    - test_thread_isolation()
    - test_tool_state_accumulation()
    """
    print("\nLangGraph Step2 Agent Loop Demo")
    print("输入 exit 退出")

    thread_id = input("请输入 thread_id，直接回车使用 demo-thread：").strip()
    if not thread_id:
        thread_id = "demo-thread"

    print(f"当前 thread_id: {thread_id}")

    while True:
        query = input("\n请输入你的问题：")

        if query.lower() in ["exit", "quit", "q"]:
            print("程序已退出")
            break

        if not query.strip():
            print("问题不能为空，请重新输入。")
            continue

        run_graph_agent(query, thread_id=thread_id)


if __name__ == "__main__":
    main_chat_loop()
