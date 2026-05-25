"""LangGraph Step 2：把手写 Agent Loop 改造成图结构。

目标：
- 理解 State / Node / Edge / Conditional Edge
- 实现 model -> tools -> model 的最小 Agent 图
- 加入 checkpoint 和 thread_id，理解有状态 Agent
- 自定义 GraphState，保存 tool_calls 和 sources
- 使用 reducer 让 model_calls / tool_calls / sources 跨轮累积
- 增加 quality_check 节点，对最终回答做规则级自检
"""
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# 当前文件在 LangGraph_learning 子目录下；
# models.py / tools.py / schemas.py 在项目根目录。
# 这里把项目根目录加入 sys.path，保证直接运行本文件时也能导入项目模块。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from agent import save_agent_log
from models import llm
from schemas import TOOLS
from tools import AVAILABLE_FUNCTIONS
from prompts import DEFAULT_SYSTEM_PROMPT

SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT

def merge_tool_calls(old: list[dict] | None, new: list[dict] | None) -> list[dict]:
    """合并工具调用记录，让 tool_calls 可以跨节点、跨轮次累积。

    LangGraph 节点每次返回新的 tool_calls 时，会调用这个 reducer。

    old：当前 state 中已有的工具调用记录
    new：当前节点新产生的工具调用记录

    返回 old + new，表示历史工具调用不会被覆盖。
    """
    return (old or []) + (new or [])


def merge_model_calls(old: list[dict] | None, new: list[dict] | None) -> list[dict]:
    """合并模型调用记录，让 model_calls 可以跨节点、跨轮次累积。"""
    return (old or []) + (new or [])


def merge_sources(old: list[str] | None, new: list[str] | None) -> list[str]:
    """合并 RAG 来源，并去重。

    sources 和 tool_calls 的区别：
    - tool_calls 通常保留每次调用的完整记录，不去重
    - sources 是文档来源，同一个文档可能多次被检索到，所以需要去重
    """
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
        使用 merge_tool_calls reducer，避免多轮工具调用时被覆盖。

    sources：
        RAG 检索来源。
        使用 merge_sources reducer，跨轮累积并去重。

    model_calls：
        结构化模型调用记录。
        使用 merge_model_calls reducer，记录每一轮模型调用耗时与工具调用情况。

    quality_check：
        最终回答的规则级自检结果。
        这个字段不加 reducer，因为它表示“当前图执行结束后的最新检查结果”，
        新结果覆盖旧结果即可。
    """

    messages: Annotated[list, add_messages]
    model_calls: Annotated[list[dict], merge_model_calls]
    tool_calls: Annotated[list[dict], merge_tool_calls]
    sources: Annotated[list[str], merge_sources]
    quality_check: dict

def call_model(state: GraphState) -> dict:
    """
    模型节点：根据当前 messages 判断是回答还是调用工具。

    输入：
    - state["messages"]：当前图状态里的对话历史

    输出：
    - {"messages": [response]}：把模型新消息追加到状态中
    """

    # 取出当前对话历史。
    # 在同一个 thread_id 下，checkpoint 会让这里拿到历史 messages。
    messages = state["messages"]
    current_round = len(state.get("model_calls", [])) + 1

    # 每次调用模型前，把 system prompt 放在最前面。
    # 区别是： LangGraph 的 messages state 里通常只放用户、模型、工具消息；
    # system prompt 可以在每次调用模型时临时拼上去。
    model_input = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    model_start_time = time.perf_counter()

    response = llm.invoke(
        model_input,
        tools=TOOLS,
        tool_choice="auto",
    )

    model_duration_ms = int((time.perf_counter() - model_start_time) * 1000)
    model_call_record = {
        "round": current_round,
        "duration_ms": model_duration_ms,
        "has_tool_calls": bool(getattr(response, "tool_calls", None)),
        "tool_call_count": len(getattr(response, "tool_calls", []) or []),
    }

    # 返回 {"messages": [response]} 后，
    # LangGraph 会使用 add_messages reducer 把 response 追加到 state["messages"] 中。
    return {
        "messages": [response],
        "model_calls": [model_call_record],
    }

def call_tools(state: GraphState) -> dict:
    """
    工具节点： 执行模型上一条消息请求调用的工具

    输入： - state["messages"][-1]: 最新模型消息，里面可能包含 tool_calls

    输出：- {"messages": tool_messages}：把工具结果追加到状态中
    """
    # 最新一条消息应该是模型消息。
    # 如果模型决定调用工具，这条消息里会带有 tool_calls。
    last_message = state["messages"][-1]

    # tool_messages：要写回 LangGraph messages 的 ToolMessage 列表。
    tool_messages = []

    # tool_call_records：结构化工具调用记录，用于最终 API 返回和调试。
    tool_call_records = []

    # new_sources：本轮工具节点新增的 RAG 来源。
    # 后续会通过 merge_sources 合并到历史 sources。
    new_sources = []

    for tool_call in last_message.tool_calls:
        function_name = tool_call["name"]
        function_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        print("\n--- LangGraph 准备执行工具 ------")
        print("工具名称：", function_name)
        print("工具参数：", function_args)

        # 工具名来自模型输出，不能假设一定存在。
        # 所以先做防御性检查。

        tool_start_time = time.perf_counter()

        if function_name not in AVAILABLE_FUNCTIONS:
            function_response = {"error": f"未知工具：{function_name}"}
        else:
            try:
                # 根据工具名找到真实 Python 函数，并展开参数执行。
                function_response = AVAILABLE_FUNCTIONS[function_name](**function_args)
            except Exception as e:
                # 工具执行失败时，把错误作为工具结果返回给模型，
                # 而不是让整个图直接崩溃。
                function_response = {"error": f"工具执行出错：{str(e)}"}
        
        tool_duration_ms = int((time.perf_counter() - tool_start_time) * 1000)
        print("工具执行结果：", function_response)
        print("工具耗时(ms)：", tool_duration_ms)

        # 保存结构化工具调用记录。
        # 这和 agent.py 里的 tool_call_records 思路一致。
        tool_call_records.append(
            {
                "name": function_name,
                "args": function_args,
                "result": function_response,
                "duration_ms": tool_duration_ms,
            }
        )
        
        # 如果调用的是 RAG 检索工具，则从检索结果中提取 source。
        # search_docs 正常返回 list[dict]；
        # 如果工具报错，可能返回 {"error": "..."}，所以要先判断类型。
        if function_name == "search_docs" and isinstance(function_response, list):
            for item in function_response:
                source = item.get("source")
                if source and source not in new_sources:
                    new_sources.append(source)

        # ToolMessage 是 LangChain/LangGraph 表示工具结果的消息类型。
        # 必须带 tool_call_id，这样模型才能知道这个结果对应哪次工具调用。
        tool_messages.append(
            ToolMessage(
                tool_call_id = tool_call_id,
                content = json.dumps(function_response,ensure_ascii=False),
            )
        )
    
    # 返回多个 state 字段：
    # - messages 会通过 add_messages 追加
    # - tool_calls 会通过 merge_tool_calls 累积
    # - sources 会通过 merge_sources 累积并去重
    return {
        "messages": tool_messages,
        "tool_calls": tool_call_records,
        "sources": new_sources,
    }

def quality_check_node(state: GraphState) -> dict:
    """最终回答质检节点。

    这个节点不调用大模型，只做规则级检查。

    为什么要加这个节点？
    - 让 LangGraph 工作流不只是 model -> tools -> model
    - 模拟真实 Agent 系统里的后处理 / 质检 / 审计节点
    - 把一些明显异常结构化返回，便于 API 层、日志和面试讲解

    当前检查项：
    - 最终回答是否为空
    - 最终回答长度
    - 是否发生过工具调用
    - 是否调用过 search_docs
    - 如果调用过 search_docs，是否记录到了 sources
    - 工具调用结果里是否存在 error
    """
    messages = state.get("messages", [])
    final_message = messages[-1] if messages else None
    answer = str(getattr(final_message, "content", "") or "")

    tool_calls = state.get("tool_calls", []) or []
    sources = state.get("sources", []) or []

    has_answer = bool(answer.strip())
    answer_length = len(answer.strip())
    has_tool_calls = bool(tool_calls)
    has_search_docs_call = any(
        tool_call.get("name") == "search_docs"
        for tool_call in tool_calls
    )
    has_sources = bool(sources)

    has_tool_error = False
    for tool_call in tool_calls:
        result = tool_call.get("result")
        if isinstance(result, dict) and result.get("error"):
            has_tool_error = True
            break

    warnings = []

    if not has_answer:
        warnings.append("最终回答为空")

    if 0 < answer_length < 10:
        warnings.append("最终回答过短，可能没有充分回答用户问题")

    if has_search_docs_call and not has_sources:
        warnings.append("调用了 search_docs，但没有记录到 sources")

    if has_tool_error:
        warnings.append("存在工具执行错误，请检查 tool_calls 中的 result")

    quality_check = {
        "has_answer": has_answer,
        "answer_length": answer_length,
        "has_tool_calls": has_tool_calls,
        "has_search_docs_call": has_search_docs_call,
        "has_sources": has_sources,
        "has_tool_error": has_tool_error,
        "warnings": warnings,
    }

    print("\n--- LangGraph Quality Check ------")
    print("quality_check:", quality_check)

    return {
        "quality_check": quality_check,
    }


def should_continue(state: GraphState) -> Literal["tools", "quality_check"]:
    """
    条件边：判断模型是否还要调用工具

    返回：
    - "tools": 如果最新模型消息里有tool_calls，下一步去工具节点
    - "quality_check"：如果没有 tool_calls，说明模型已经给出最终回答，下一步进入质检节点
    """
    last_message = state["messages"][-1]

    # 如果最新模型消息里有 tool_calls，说明模型还需要工具结果。
    # 下一步进入 tools 节点。
    if getattr(last_message, "tool_calls", None):
        return "tools"
    
    # 如果没有 tool_calls，说明模型已经给出最终回答。
    # 不直接结束，而是进入 quality_check 节点做规则级自检。
    return "quality_check"

def build_graph():
    """
    构建并编译 LangGraph Agent 图。

    图结构：
    START → model → 条件判断
                    ├── tools → model
                    └── quality_check → END
    """
    # StateGraph(GraphState) 表示整张图共享 GraphState 这个状态结构。
    workflow = StateGraph(GraphState)

    # 添加节点。
    # 节点名是图中的名字，可以任意命名；这里沿用 "models" / "tools"。
    workflow.add_node("models", call_model)
    workflow.add_node("tools", call_tools)
    workflow.add_node("quality_check", quality_check_node)

    # 从 START 进入 model节点
    workflow.add_edge(START, "models")

    # model 节点执行完后，根据 should_continue 的返回值选择下一步
    workflow.add_conditional_edges(
        "models",
        should_continue,
        {
            "tools": "tools",
            "quality_check": "quality_check",
        },
    )

    # tools 执行完后，回到model，让模型基于工具结果继续判断或回答
    workflow.add_edge("tools", "models")

    # 最终回答生成后，进入 quality_check 节点；
    # 质检完成后，整张图结束。
    workflow.add_edge("quality_check", END)
    
    # InMemorySaver 表示状态保存在当前 Python 进程内存里
    # 特点：适合学习和demo，程序一关，状态消失
    # 严格来说，这个“退出再启动”的测试，不能验证跨进程持久化
    checkpointer = InMemorySaver()

    # 编译图，并绑定 checkpointer。
    # 之后 invoke 时只要传入 thread_id，就可以保存和读取对应会话状态。
    return workflow.compile(checkpointer=checkpointer)

graph_app = build_graph()


def run_graph_agent(user_query: str, thread_id: str = "demo-thread") -> dict:
    """运行 LangGraph Agent。

    参数：
    - user_query：用户问题
    - thread_id：LangGraph 会话 ID，用于 checkpoint 区分不同对话线程

    返回：
    {
        "thread_id": "当前会话 ID",
        "answer": "最终回答",
        "tool_calls": "累计工具调用记录",
        "sources": "累计 RAG 来源",
        "messages_count": "当前会话累计消息数量"
    }
    """
    # trace_id 是本次 LangGraph Agent 请求的唯一编号。
    # 注意：trace_id 是“请求级别”的，不放进 GraphState，
    # 避免被 checkpoint 当成会话状态长期保存。
    trace_id = uuid.uuid4().hex

    # start_time 用来统计本次 run_graph_agent 调用的总耗时。
    start_time = time.perf_counter()

    # 初始输入 state。
    # messages 放当前用户消息；
    # tool_calls / sources 初始为空列表。
    # 如果同一个 thread_id 已有历史，checkpoint 会自动合并历史状态。
    inputs = {
        "messages": [
            {
                "role": "user",
                "content": user_query,
            }
        ],
        "model_calls": [],
        "tool_calls": [],
        "sources": [],
        "quality_check": {},
    }

    # configurable.thread_id 是 LangGraph checkpointer 识别会话的关键。
    # 同一个 thread_id 会读取历史状态；
    # 不同 thread_id 会互相隔离。
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    # 运行整张图，直到 END。
    final_state = graph_app.invoke(inputs, config=config)

    # 图结束时最后一条消息通常就是模型最终回答。
    final_message = final_state["messages"][-1]

    duration_ms = int((time.perf_counter() - start_time) * 1000)

    # messages_count：当前会话累计消息数量
    # 这个字段可以帮助我观察 checkpoint 是否在累积历史
    result = {
        "trace_id": trace_id,
        "agent_type": "langgraph",
        "thread_id": thread_id,
        "answer": final_message.content,
        "model_calls": final_state.get("model_calls", []),
        "tool_calls": final_state.get("tool_calls", []),
        "sources": final_state.get("sources", []),
        "quality_check": final_state.get("quality_check", {}),
        "messages_count": len(final_state["messages"]),
        "duration_ms": duration_ms,
        "success": True,
        "error": None,
    }

    print("\nLangGraph 最终回答：")
    print(result["answer"])
    print("工具调用：", result["tool_calls"])
    print("来源：", result["sources"])
    print("质检：", result["quality_check"])
    print("messages_count：", result["messages_count"])

    save_agent_log(result)

    return result

def test_thread_isolation() -> None:
    """测试不同 thread_id 之间的会话隔离。

    预期：
    - user-a 先告诉 Agent 自己叫小明
    - user-b 询问名字时，不应该知道小明
    - user-a 再询问名字时，应该知道小明
    """
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
    """测试 tool_calls 和 sources 是否会在同一个 thread_id 下累积。

    预期：
    - 第一轮计算问题产生 calculator 工具调用
    - 第二轮 RAG 问题产生 search_docs 工具调用
    - 第二轮返回的 tool_calls 中应该同时包含 calculator 和 search_docs
    """
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
