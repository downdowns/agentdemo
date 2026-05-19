"""FastAPI 服务入口。

这个文件负责把命令行里的 Agent 能力封装成 HTTP API。

当前提供两个接口：
- GET /health：健康检查，用来确认服务是否启动成功。
- POST /chat：聊天接口，接收用户问题并调用 run_agent() 返回结构化结果。

启动方式：
    uvicorn app.main:app --reload
"""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# app/main.py 位于 app 目录下，而 agent.py 位于项目根目录。
# 为了让当前文件能稳定导入 agent.py，这里把项目根目录加入 Python 模块搜索路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# run_agent 是整个 Agent 的核心入口：
# FastAPI 接口收到请求后，会把用户问题交给 run_agent 处理。
from agent import run_agent


class ChatRequest(BaseModel):
    """聊天请求体。

    客户端请求 /chat 时，需要传入：
    {
        "message": "用户问题"
    }

    Pydantic 会自动校验 message 字段是否存在、是否是字符串。
    """

    message: str


# 创建 FastAPI 应用对象。
# title / description / version 会展示在 Swagger 文档中。
app = FastAPI(
    title="Enterprise RAG Agent API",
    description="A FastAPI service for RAG + Tool Calling Agent.",
    version="0.1.0",
)


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
def chat(request: ChatRequest) -> dict:
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
