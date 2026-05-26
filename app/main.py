"""FastAPI 服务入口。

这个文件负责把命令行里的 Agent 能力封装成 HTTP API。

当前提供四个接口：
- GET /health：健康检查，用来确认服务是否启动成功。
- POST /chat：聊天接口，接收用户问题并调用 run_agent() 返回结构化结果。
- POST /chat/stream：流式聊天接口，使用 SSE 结构化事件逐段返回最终回答。
- POST /chat/langgraph：LangGraph Agent 接口，支持 session_id 有状态会话。

启动方式：
    uvicorn app.main:app --reload
"""

import json
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


# app/main.py 位于 app 目录下，而 agent.py 位于项目根目录。
# 为了让当前文件能稳定导入 agent.py，这里把项目根目录加入 Python 模块搜索路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# run_agent 是整个 Agent 的核心入口：
# FastAPI 接口收到请求后，会把用户问题交给 run_agent 处理。
from agent_workflows.langgraph_agent import run_graph_agent
from agent import run_agent
from database import (
    save_agent_trace,
    save_chat_message,
    save_chat_session,
    save_tool_call_log,
)


class ChatRequest(BaseModel):
    """聊天请求体。

    客户端请求 /chat 时，需要传入：
    {
        "message": "用户问题"
    }

    Pydantic 会自动校验 message 字段是否存在、是否是字符串。
    """

    message: str


class LangGraphChatRequest(BaseModel):
    """LangGraph 聊天请求体。

    客户端请求 /chat/langgraph 时，需要传入：
    {
        "message": "用户问题",
        "session_id": "user-a"
    }

    session_id 是 API 层的会话 ID。
    在接口内部会映射为 LangGraph 的 thread_id：
        session_id -> thread_id

    同一个 session_id 可以保留上下文；
    不同 session_id 之间状态隔离。
    """

    message: str
    session_id: str = "demo-thread"


# 创建 FastAPI 应用对象。
# title / description / version 会展示在 Swagger 文档中。
app = FastAPI(
    title="Enterprise RAG Agent API",
    description="A FastAPI service for RAG + Tool Calling Agent.",
    version="0.1.0",
)

RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "20"))

RATE_LIMIT_BUCKETS: dict[str, list[float]] = {}

def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """校验 API Key。

    客户端需要在请求头中传：
        X-API-Key: xxx

    服务端从 .env 读取 APP_API_KEY。
    如果未配置或不匹配，则拒绝请求。
    """

    expected_api_key = os.getenv("APP_API_KEY")

    if not expected_api_key:
        raise HTTPException(
            status_code=500,
            detail="服务端未配置 APP_API_KEY",
        )

    if x_api_key != expected_api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key",
        )
    
    check_rate_limit(x_api_key)

def check_rate_limit(api_key: str) -> None:
    """检查 API Key 是否超过请求频率限制。

    当前实现是单进程内存版滑动窗口：
    - 每个 API Key 对应一个请求时间列表；
    - 每次请求时清理窗口外的旧时间；
    - 如果窗口内请求次数达到上限，则返回 429。
    """

    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    request_times = RATE_LIMIT_BUCKETS.get(api_key, [])

    # 只保留当前窗口内的请求时间。
    request_times = [
        request_time
        for request_time in request_times
        if request_time >= window_start
    ]

    if len(request_times) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Too Many Requests",
        )

    request_times.append(now)
    RATE_LIMIT_BUCKETS[api_key] = request_times

@app.get("/health")
def health_check() -> dict:
    """健康检查接口。

    这个接口不调用模型，也不调用向量库，只用于确认 FastAPI 服务是否正常运行。
    """
    return {
        "status": "ok",
        "service": "rag-agent-api",
    }


@app.post("/chat")
def chat(
    request: ChatRequest,
    _: None = Depends(verify_api_key),
) -> dict:
    """Agent 聊天接口。

    流程：
    1. 从请求体中取出用户问题 request.message
    2. 调用 run_agent()
    3. 直接返回 Agent 的结构化结果

    当前返回字段包括：
    - user_query：用户原始问题
    - answer：模型最终回答
    - tool_calls：工具调用记录
    - sources：RAG 检索来源
    - rounds：Agent 运行轮数
    """
    # 空 message 校验(请求校验)，解决了用户传空问题
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="message 不能为空",
        )

    # Agent 异常捕获，这让 Agent 内部异常不会直接变成不可控报错，而是变成规范 HTTP 响应。
    try:
        result = run_agent(request.message)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent 执行失败：{str(e)}",
        )


def stream_agent_result_sse(result: dict, chunk_size: int = 20):
    """把 Agent 返回结果转换成 SSE 流式事件。

    SSE 每条消息格式：
        data: JSON字符串\n\n

    当前事件类型：
    - metadata：请求元信息，例如 trace_id、sources、duration_ms
    - answer_delta：答案片段
    - error：流式输出过程中的错误
    - done：流式输出结束
    """
    try:
        metadata_event = {
            "type": "metadata",
            "trace_id": result.get("trace_id"),
            "sources": result.get("sources", []),
            "duration_ms": result.get("duration_ms"),
            "success": result.get("success"),
        }

        yield f"data: {json.dumps(metadata_event, ensure_ascii=False)}\n\n"

        answer = str(result.get("answer", ""))

        for i in range(0, len(answer), chunk_size):
            chunk = answer[i : i + chunk_size]

            event = {
                "type": "answer_delta",
                "content": chunk,
            }

            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            # 这里 sleep 只是为了本地测试时更明显地看到“流式效果”。
            # 真实生产中不一定需要人为 sleep。
            time.sleep(0.03)

        done_event = {
            "type": "done",
            "trace_id": result.get("trace_id"),
        }

        yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"

    except Exception as e:
        error_event = {
            "type": "error",
            "message": str(e),
            "trace_id": result.get("trace_id"),
        }

        yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

        done_event = {
            "type": "done",
            "trace_id": result.get("trace_id"),
        }

        yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
def chat_stream(
    request: ChatRequest,
    _: None = Depends(verify_api_key),
):
    """Agent 流式输出接口。

    注意：
    这是简单版 streaming：
    1. 先调用 run_agent() 得到完整结构化结果
    2. 再把 result["answer"] 按小段流式返回

    它主要用于学习 FastAPI StreamingResponse。
    后续可以升级为真正的模型 token streaming。
    """
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="message 不能为空",
        )

    try:
        result = run_agent(request.message)

        return StreamingResponse(
            stream_agent_result_sse(result),
            media_type="text/event-stream",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent 流式执行失败：{str(e)}",
        )


@app.post("/chat/langgraph")
def chat_langgraph(
    request: LangGraphChatRequest,
    _: None = Depends(verify_api_key),
) -> dict:
    """LangGraph Agent 聊天接口。

    和 /chat 的区别：
    - /chat 调用手写 Agent Loop
    - /chat/langgraph 调用 LangGraph StateGraph 版本

    LangGraph 版本支持：
    - checkpoint
    - thread_id/session_id
    - 有状态多轮对话
    - 多会话隔离
    - tool_calls / sources 跨轮累积
    """
    message = request.message.strip()
    session_id = request.session_id.strip()

    # 校验用户问题不能为空。
    if not message:
        raise HTTPException(
            status_code=400,
            detail="message 不能为空",
        )

    # session_id 不能为空。
    # 如果为空，LangGraph checkpointer 无法可靠地区分会话。
    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id 不能为空",
        )

    try:
        # API 层叫 session_id，更符合用户/前端的理解；
        # LangGraph 层叫 thread_id，是 checkpointer 用来区分状态的字段。
        # 这里完成 session_id -> thread_id 的映射。
        result = run_graph_agent(
            user_query=message,
            thread_id=session_id,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LangGraph Agent 执行失败：{str(e)}",
        )

    try:
        # 这里保存数据库
        trace_id = result.get("trace_id")
        answer = result.get("answer", "")

        save_chat_session(
            session_id=session_id,
            agent_type="langgraph",
        )

        save_chat_message(
            session_id=session_id,
            trace_id=trace_id,
            role="user",
            content=message,
        )

        save_chat_message(
            session_id=session_id,
            trace_id=trace_id,
            role="assistant",
            content=answer,
        )

        save_agent_trace(
            trace_id=trace_id,
            session_id=session_id,
            user_query=message,
            answer=answer,
            sources=result.get("sources", []),
            model_calls=result.get("model_calls", []),
            quality_check=result.get("quality_check", {}),
            duration_ms=result.get("duration_ms"),
            success=result.get("success", True),
            error=result.get("error"),
            agent_type="langgraph",
        )

        for tool_call in result.get("tool_calls", []):
            save_tool_call_log(
                trace_id=trace_id,
                session_id=session_id,
                tool_name=tool_call.get("name", "unknown"),
                tool_args=tool_call.get("args"),
                tool_result=tool_call.get("result"),
                duration_ms=tool_call.get("duration_ms"),
                success=tool_call.get("success", True),
                error=tool_call.get("error"),
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LangGraph Agent 数据库保存失败：{str(e)}",
        )
    
    return result