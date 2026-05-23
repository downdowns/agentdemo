# FastAPI 请求校验、错误处理与大模型服务化

> 本文是学习型知识库文档，基于 FastAPI 官方 Request Body 文档和本项目实践整理，用于企业知识库 Agent API 服务化实验。

## 1. 为什么 RAG Agent 需要 FastAPI

本地命令行 Demo 只能证明 Agent 能跑，但真实项目需要通过 HTTP 接口提供服务。FastAPI 的作用是把 Python 中的 Agent 函数封装成可调用的 Web API。

在本项目中，FastAPI 提供了两个主要接口：

```text
GET  /health
POST /chat
POST /chat/langgraph
```

其中：

- `/health` 用于健康检查；
- `/chat` 调用手写 Function Calling Agent Loop；
- `/chat/langgraph` 调用 LangGraph Agent，并支持 session_id / thread_id 的有状态会话。

通过 FastAPI，前端、测试脚本或其他服务都可以通过 HTTP 请求调用 Agent。

## 2. Pydantic 请求模型

FastAPI 常用 Pydantic 模型定义请求体。比如：

```python
class ChatRequest(BaseModel):
    question: str
```

接口函数可以写成：

```python
@app.post("/chat")
def chat(request: ChatRequest):
    return run_agent(request.question)
```

这样 FastAPI 会自动完成：

1. 解析 JSON 请求体；
2. 校验字段是否存在；
3. 校验字段类型是否正确；
4. 生成 OpenAPI 文档；
5. 在 Swagger UI 中展示请求结构。

如果用户没有传 question，或者 question 类型不是字符串，FastAPI 会返回校验错误。

## 3. 为什么还要做业务校验

Pydantic 可以校验类型，但业务规则通常还需要自己补充。例如 question 虽然是字符串，但可能是空字符串：

```json
{"question": ""}
```

这种请求类型合法，但业务上没有意义。因此本项目在接口层增加了空问题校验：

```python
if not request.question.strip():
    raise HTTPException(status_code=400, detail="question 不能为空")
```

这属于业务校验。它能让接口返回更明确的错误信息，也能避免 Agent 收到无效输入。

## 4. HTTPException 的作用

`HTTPException` 用来主动返回 HTTP 错误响应。例如：

```python
raise HTTPException(status_code=400, detail="question 不能为空")
```

含义是：

- `400` 表示客户端请求有问题；
- `detail` 是返回给调用方的错误说明。

如果 Agent 执行过程中发生内部异常，也可以捕获后返回 `500`，表示服务端错误。但开发阶段也需要保留日志，方便排查真实原因。

## 5. 结构化返回为什么重要

如果接口只返回一个字符串，前端和调试工具很难知道 Agent 内部发生了什么。本项目采用结构化返回，例如：

```json
{
  "answer": "RAG 是检索增强生成...",
  "tool_calls": [
    {"name": "search_docs", "args": {"query": "RAG 是什么"}}
  ],
  "sources": ["langchain_rag.md"],
  "rounds": 2
}
```

结构化返回的好处是：

1. 前端可以展示答案和来源；
2. 日志可以记录工具调用轨迹；
3. 评估脚本可以提取 tool_calls 和 sources；
4. 面试时可以说明系统具备可观测性；
5. 后续可以扩展 trace_id、latency、token_usage 等字段。

## 6. Swagger 调试

FastAPI 默认提供 Swagger UI，一般访问：

```text
http://127.0.0.1:8000/docs
```

在 Swagger 页面中，可以直接填写请求 JSON 并测试接口。这对本地开发非常方便，尤其是验证：

- 正常请求返回 200；
- 空 question 返回 400；
- `/chat/langgraph` 是否保留 session 记忆；
- 不同 session_id 是否隔离；
- Agent 是否返回 tool_calls 和 sources。

## 7. 大模型应用服务化的注意点

把 RAG Agent 封装成 API 后，还需要考虑更多工程问题：

1. **超时控制**：模型调用和检索可能较慢，需要设置超时。
2. **异常处理**：工具异常不能导致服务崩溃，要返回可理解错误。
3. **日志记录**：记录 question、answer、tool_calls、sources、latency。
4. **鉴权**：企业知识库通常需要用户身份和权限控制。
5. **限流**：避免接口被恶意或高频调用。
6. **会话管理**：session_id / thread_id 需要持久化和过期策略。
7. **部署**：生产环境通常需要 Docker、进程管理、反向代理和监控。

本项目目前完成了请求校验、异常处理、结构化返回和基础日志，后续可以继续补充 Docker、鉴权和持久化 checkpoint。

## 8. 面试回答模板

如果面试官问“FastAPI 在你的项目中起什么作用”，可以回答：

> FastAPI 负责把本地 Agent 封装成 HTTP 服务。我定义了请求模型进行参数校验，对空问题返回 400 错误，并把 Agent 的 answer、tool_calls、sources 等信息结构化返回。这样项目不只是命令行 Demo，而是可以被前端或其他服务调用的后端接口。同时 Swagger 也方便调试 `/chat` 和 `/chat/langgraph`。

## 参考来源

- FastAPI Request Body 文档：https://fastapi.tiangolo.com/tutorial/body/
