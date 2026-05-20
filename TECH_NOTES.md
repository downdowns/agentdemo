# RAG Agent 技术知识点整理

> 这份文档用于补足基础知识。你当前主要通过实操学习，这里把项目里用到的概念系统整理一遍，方便复习和面试表达。

---

## 1. LLM 应用的几种形态

### 1.1 普通 Chatbot

流程：

```text
用户问题 → LLM → 答案
```

特点：

- 实现简单
- 依赖模型自身知识
- 容易出现幻觉
- 无法访问外部实时信息或本地私有知识

---

### 1.2 RAG

RAG 全称：

```text
Retrieval-Augmented Generation
```

中文：

```text
检索增强生成
```

流程：

```text
用户问题
  ↓
检索知识库
  ↓
把相关文档片段放进 Prompt
  ↓
LLM 基于文档回答
```

RAG 解决的问题：

- 让模型回答本地私有知识
- 降低幻觉
- 提供答案来源
- 不需要频繁重新训练模型

---

### 1.3 Agent

Agent 可以理解为：

```text
LLM + Tools + Memory/State + Loop/Workflow
```

最小 Agent Loop：

```text
用户问题
  ↓
LLM 判断是否需要工具
  ↓
如果需要，生成 tool_calls
  ↓
程序执行真实工具
  ↓
工具结果返回给 LLM
  ↓
LLM 继续判断或生成最终答案
```

Agent 和普通 Chatbot 的区别：

| 对比项 | Chatbot | Agent |
|---|---|---|
| 是否能用工具 | 通常不能 | 可以 |
| 流程是否固定 | 固定 | 模型可决策 |
| 是否能访问外部系统 | 较弱 | 较强 |
| 风险 | 幻觉 | 幻觉 + 工具误用 |

---

## 2. Function Calling / Tool Calling

Function Calling 是让模型输出结构化工具调用请求。

模型不是直接执行 Python 函数，而是输出：

```json
{
  "name": "calculator",
  "args": {
    "operation": "multiply",
    "a": 23,
    "b": 19
  }
}
```

程序拿到这个结果后，再执行真实函数：

```python
calculator(operation="multiply", a=23, b=19)
```

### 2.1 工具调用的关键组成

1. 工具真实函数：给程序执行
2. 工具 schema：给模型阅读
3. 工具映射表：把模型返回的工具名映射到 Python 函数
4. Agent Loop：控制模型调用和工具执行

---

### 2.2 为什么需要 tool_call_id？

模型可能一次调用多个工具。

例如：

```text
查天气 + 做计算
```

每个工具调用都有一个 `tool_call_id`，程序把工具结果返回给模型时必须带上它。

作用：

```text
让模型知道哪个工具结果对应哪个工具请求。
```

---

### 2.3 为什么工具结果要返回给模型？

工具执行结果只是程序知道，模型不知道。

所以必须把结果作为 `tool` 消息追加到 `messages`：

```python
messages.append({
    "role": "tool",
    "tool_call_id": tool_call_id,
    "content": json.dumps(function_response, ensure_ascii=False),
})
```

然后下一轮再调用模型，模型才能基于工具结果回答。

---

## 3. RAG 核心流程

### 3.1 文档加载

把本地 `.md` / `.txt` 文件读取出来，包装成 LangChain `Document`。

Document 通常包含：

```text
page_content：正文
metadata：元数据，例如 source
```

---

### 3.2 文档切分

为什么要切分？

1. 长文档不能全部塞进 prompt
2. 向量检索通常以片段为单位
3. 小片段更容易匹配用户问题

常见参数：

```python
chunk_size = 100
chunk_overlap = 20
```

`chunk_size`：每个片段大概多长。  
`chunk_overlap`：相邻片段重叠多少，避免语义被切断。

---

### 3.3 Embedding

Embedding 是把文本转成向量。

例如：

```text
"RAG 是什么" → [0.12, -0.03, 0.88, ...]
```

向量之间可以计算相似度，语义越接近，相似度越高。

当前项目使用：

```text
BAAI/bge-m3
```

它适合中英文语义检索。

---

### 3.4 向量数据库 Chroma

Chroma 用来存储：

1. 文档片段
2. 文档向量
3. metadata

检索时：

```text
用户问题 → embedding → 向量相似度搜索 → top-k 文档片段
```

---

### 3.5 top-k

`k` 表示返回最相关的前几个片段。

例如：

```python
similarity_search(query, k=2)
```

返回最相似的 2 个片段。

`k` 太小：

- 可能漏掉答案

`k` 太大：

- prompt 变长
- 噪声变多
- 成本增加

---

## 4. RAG + Agent 的结合

普通 RAG 是固定流程：

```text
用户问题 → 必定检索 → LLM 回答
```

Agent 化 RAG 是：

```text
用户问题 → 模型判断是否需要检索 → search_docs → LLM 回答
```

也就是说：

```text
RAG 检索能力被封装成一个工具。
```

好处：

1. 普通聊天不必检索
2. 计算问题不必检索
3. 复杂问题可以同时检索和调用其他工具
4. 更接近真实 Agent 系统

---

## 5. 结构化返回

项目级 Agent 不能只 `print()`，要 `return dict`。

当前返回：

```python
{
    "user_query": "...",
    "answer": "...",
    "tool_calls": [...],
    "sources": [...],
    "rounds": 2
}
```

作用：

- FastAPI 可以直接返回 JSON
- 前端可以展示
- 日志可以保存
- 评估脚本可以读取
- 面试时能展示 Agent 执行过程

---

## 6. 工具异常处理

工具可能失败：

- 参数错误
- 网络错误
- 向量库错误
- 外部 API 错误

所以要用：

```python
try:
    function_response = tool(**args)
except Exception as e:
    function_response = {"error": str(e)}
```

这样做的好处：

```text
工具失败不会导致整个 Agent 崩溃。
```

模型还能基于错误信息生成友好回答。

---

## 7. 最大轮数限制

Agent 可能陷入循环：

```text
模型 → 工具 → 模型 → 工具 → ...
```

所以要设置：

```python
MAX_AGENT_ROUNDS = 5
```

如果超过最大轮数，直接停止。

这是 Agent 安全性和稳定性的基本措施。

---

## 8. 日志 JSONL

当前项目把每次 Agent 运行结果写入：

```text
logs/agent.log
```

每一行是一个 JSON：

```json
{"user_query": "...", "answer": "...", "tool_calls": [...], "sources": [...], "rounds": 2}
```

这种格式叫 JSONL：

```text
JSON Lines
```

优点：

- 追加写入简单
- 一行一条记录
- 方便后续分析
- 方便做离线评估

---

## 9. Agent Evaluation

Agent 测评的目标：

```text
判断 Agent 有没有按预期工作。
```

当前项目先做最小评估：

```text
工具调用是否正确
```

流程：

```text
读取问题集
  ↓
调用 run_agent
  ↓
提取 actual_tools
  ↓
对比 expected_tools
  ↓
统计通过率
```

当前评估指标：

```text
工具调用通过率
```

后续可以扩展：

1. 工具参数是否正确
2. RAG 来源是否命中
3. 答案是否包含关键事实
4. 是否出现幻觉
5. 响应时间
6. token 成本

---

## 10. FastAPI 服务化

FastAPI 的作用：

```text
把 Python 函数封装成 HTTP API。
```

当前接口：

```text
GET /health
POST /chat
```

为什么要服务化？

1. 前端可以调用
2. 其他系统可以调用
3. 方便部署
4. 方便演示
5. 更接近真实项目

---

## 11. Pydantic

FastAPI 用 Pydantic 定义请求体：

```python
class ChatRequest(BaseModel):
    message: str
```

作用：

- 自动解析 JSON
- 自动校验字段
- 自动生成接口文档
- 请求格式错误时自动返回 422

---

## 12. Swagger 文档

FastAPI 自动生成 Swagger 文档：

```text
http://127.0.0.1:8000/docs
```

用途：

- 查看接口
- 在线调试
- 演示项目
- 给面试官展示服务化能力

---

## 13. 当前技术栈总结

| 模块 | 技术 |
|---|---|
| LLM 调用 | LangChain `ChatOpenAI` |
| 模型 API | DeepSeek OpenAI-compatible API |
| Embedding | `BAAI/bge-m3` |
| 文档结构 | LangChain `Document` |
| 文档切分 | `RecursiveCharacterTextSplitter` |
| 向量库 | Chroma |
| Agent | 手写 Function Calling Agent Loop |
| 工具定义 | JSON Schema |
| API 服务 | FastAPI |
| 请求校验 | Pydantic |
| 评估 | 自定义 eval 脚本 |
| 日志 | JSONL |

---

## 14. LangGraph 入门知识

你目前已经把手写 Agent Loop 改写成了 LangGraph 学习版。

### 14.1 为什么要学 LangGraph

手写 Agent Loop 适合简单场景：

```text
model → tools → model
```

但当流程变复杂时，例如：

```text
router → rag → rerank → generate → verify → human_review
```

如果全部用 `if/else/while` 手写，代码会越来越难维护。

LangGraph 的价值是：

```text
把 Agent 流程显式建模成 State + Node + Edge。
```

---

### 14.2 手写 Agent Loop 和 LangGraph 的对应关系

| 手写 Agent Loop | LangGraph |
|---|---|
| `messages` | `MessagesState` |
| 调用模型 | model node |
| 执行工具 | tools node |
| `if message.tool_calls` | conditional edge |
| 循环调用模型 | tools → model 回边 |
| 手动管理历史 | checkpointer |
| 会话 ID | thread_id |

---

### 14.3 State

State 是图运行时保存的数据。

你当前使用：

```python
MessagesState
```

可以理解成：

```python
state = {
    "messages": [...]
}
```

---

### 14.4 Node

Node 是图中的一个处理步骤，本质上就是 Python 函数。

你当前有：

```text
call_model：调用模型
call_tools：执行工具
```

节点输入 state，返回新的 state 片段：

```python
return {"messages": [response]}
```

---

### 14.5 Conditional Edge

Conditional Edge 是条件边。

你当前的判断逻辑：

```python
if last_message.tool_calls:
    return "tools"
return END
```

含义：

```text
如果模型要调用工具 → 去 tools 节点
如果模型不调用工具 → 结束图
```

---

## 15. LangGraph checkpoint 和 thread_id

### 15.1 checkpoint 是什么

checkpoint 用来保存图运行状态。

你当前保存的主要是：

```python
{
    "messages": [...]
}
```

没有 checkpoint 时：

```text
每次调用只看当前输入
```

有 checkpoint 后：

```text
同一个 thread_id 可以延续历史 messages
```

---

### 15.2 InMemorySaver

你当前使用：

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
workflow.compile(checkpointer=checkpointer)
```

它表示：

```text
把 checkpoint 存在当前 Python 进程内存里。
```

适合：

- 学习
- demo
- 临时测试

不适合：

- 程序重启后还要保留状态的生产环境

---

### 15.3 thread_id 是什么

`thread_id` 是会话 ID。

调用 LangGraph 时传：

```python
config = {
    "configurable": {
        "thread_id": thread_id,
    }
}

graph_app.invoke(inputs, config=config)
```

作用：

```text
告诉 LangGraph 当前这次调用属于哪个会话。
```

---

### 15.4 同一个 thread_id 如何记住历史

示例：

```python
run_graph_agent("我叫小明", thread_id="user-a")
run_graph_agent("我叫什么？", thread_id="user-a")
```

因为两次都是 `user-a`，所以第二次能读取第一次保存的 messages。

---

### 15.5 不同 thread_id 如何隔离

示例：

```python
run_graph_agent("我叫小明", thread_id="user-a")
run_graph_agent("我叫什么？", thread_id="user-b")
run_graph_agent("我叫什么？", thread_id="user-a")
```

预期：

```text
user-b 不知道小明
user-a 知道小明
```

这证明不同 `thread_id` 的状态互相隔离。

---

### 15.6 messages_count 怎么理解

`messages_count` 是当前会话累计消息条数，不是对话轮数。

普通一轮对话：

```text
user → assistant
```

增加 2 条消息。

调用工具的一轮：

```text
user → assistant(tool_call) → tool → assistant(final)
```

增加 4 条消息。

多个工具会增加更多 tool message。

---

## 16. LangGraph 自定义 State 与 reducer

### 16.1 为什么要自定义 State

一开始你使用的是：

```python
MessagesState
```

它适合只保存：

```python
{
    "messages": [...]
}
```

但项目化 Agent 通常不只需要 messages，还需要保存业务字段，例如：

```text
tool_calls
sources
retrieved_docs
final_answer
session_id
```

所以你定义了自定义 State：

```python
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls: Annotated[list[dict], merge_tool_calls]
    sources: Annotated[list[str], merge_sources]
```

---

### 16.2 Annotated 是什么

在 LangGraph 中：

```python
messages: Annotated[list, add_messages]
```

表示：

```text
messages 是 list 类型，并且使用 add_messages 这个 reducer 来合并新旧状态。
```

`Annotated` 的作用是给字段附加额外元信息。

---

### 16.3 reducer 是什么

reducer 是“状态合并函数”。

当节点返回：

```python
{"tool_calls": new_tool_calls}
```

LangGraph 需要知道：

```text
新 tool_calls 是覆盖旧值？
还是追加到旧值后面？
```

reducer 就是用来定义这个规则的。

---

### 16.4 add_messages

`add_messages` 是 LangGraph 内置 reducer。

它的作用：

```text
把新 messages 追加到旧 messages 后面。
```

所以模型消息、工具消息、用户消息可以持续累积。

---

### 16.5 merge_tool_calls

你自定义了：

```python
def merge_tool_calls(old: list[dict] | None, new: list[dict] | None) -> list[dict]:
    return (old or []) + (new or [])
```

作用：

```text
旧工具调用记录 + 新工具调用记录
```

这样同一个 `thread_id` 下，历史工具调用不会被覆盖。

---

### 16.6 merge_sources

你自定义了：

```python
def merge_sources(old: list[str] | None, new: list[str] | None) -> list[str]:
    merged = []

    for source in (old or []) + (new or []):
        if source and source not in merged:
            merged.append(source)

    return merged
```

作用：

```text
累积 RAG 来源，并去重。
```

为什么要去重？

因为同一个文档可能被多轮 RAG 检索重复命中。

---

### 16.7 不加 reducer 会怎样

如果写成：

```python
tool_calls: list[dict]
sources: list[str]
```

节点返回新值时，可能会覆盖旧值。

这意味着：

```text
第一轮 calculator 记录可能被第二轮 search_docs 记录覆盖。
```

加 reducer 后，就可以跨多轮累积。

---

### 16.8 当前 LangGraph 学习版返回结构

当前 `run_graph_agent()` 返回：

```python
{
    "thread_id": thread_id,
    "answer": final_message.content,
    "tool_calls": final_state.get("tool_calls", []),
    "sources": final_state.get("sources", []),
    "messages_count": len(final_state["messages"]),
}
```

这已经接近主项目 `agent.py` 的结构化返回。

---

### 16.9 当前测试函数

你现在有两个测试函数：

```python
test_thread_isolation()
test_tool_state_accumulation()
```

`test_thread_isolation()` 验证：

```text
不同 thread_id 不共享记忆。
```

`test_tool_state_accumulation()` 验证：

```text
同一个 thread_id 下，tool_calls 和 sources 可以跨轮累积。
```

---

## 17. LangGraph Agent 服务化

### 17.1 为什么要把 LangGraph 接入 FastAPI

命令行版本适合学习和调试，但真实项目需要通过 HTTP API 调用。

因此你新增了：

```text
POST /chat/langgraph
```

这个接口调用：

```python
run_graph_agent()
```

---

### 17.2 session_id 和 thread_id 的关系

API 层使用：

```text
session_id
```

LangGraph 层使用：

```text
thread_id
```

在接口中做映射：

```python
result = run_graph_agent(
    user_query=request.message,
    thread_id=request.session_id,
)
```

也就是说：

```text
session_id → thread_id
```

---

### 17.3 /chat 和 /chat/langgraph 的区别

| 接口 | 实现 | 特点 |
|---|---|---|
| `/chat` | 手写 Agent Loop | 适合理解底层 Function Calling |
| `/chat/langgraph` | LangGraph Agent | 支持 checkpoint、session_id、有状态会话 |

---

### 17.4 /chat/langgraph 请求示例

```json
{
  "message": "我叫什么？",
  "session_id": "user-a"
}
```

返回：

```json
{
  "thread_id": "user-a",
  "answer": "你叫小明。",
  "tool_calls": [],
  "sources": [],
  "messages_count": 4
}
```

---

### 17.5 /chat/langgraph 已验证能力

你已经测试通过：

1. 同一个 `session_id` 能记住历史
2. 不同 `session_id` 之间状态隔离
3. LangGraph 接口能正常调用工具
4. 空 `message` 返回 HTTP 400
5. 空 `session_id` 返回 HTTP 400
