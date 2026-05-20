# Enterprise RAG Agent 项目复习总览

> 这份文档是“复习版项目说明”。  
> 目标不是展示给别人看，而是帮你自己重新理清：这个项目到底做了什么、每个文件干什么、一次请求怎么流转、你目前学到了哪些知识。

---

## 0. 一句话总结这个项目

这是一个基于 **RAG + Function Calling + Agent Loop + FastAPI** 的多工具知识库 Agent。

它可以让模型根据用户问题自动决定是否调用工具：

- 问知识库内容 → 调用 `search_docs`
- 问数学计算 → 调用 `calculator`
- 问天气 → 调用 `get_weather`
- 混合问题 → 可以一次调用多个工具

工具执行完成后，结果会返回给模型，模型再基于工具结果生成最终回答。

---

## 1. 你目前已经完成了什么

当前项目已经完成：

1. RAG 本地知识库
2. 文档加载与切分
3. Embedding 向量化
4. Chroma 向量库
5. `search_docs` 检索工具
6. `calculator` 计算工具
7. `get_weather` 模拟天气工具
8. 工具 Schema
9. 手写 Function Calling Agent Loop
10. 结构化返回
11. 工具调用记录
12. RAG 来源追踪
13. 工具异常处理
14. 最大轮数限制
15. JSONL 日志
16. 最小 Agent 评估脚本
17. FastAPI `/health` 和 `/chat`
18. FastAPI 空输入校验
19. FastAPI 异常处理
20. README / 技术文档 / 面试文档
21. GitHub 上传
22. LangGraph 最小 Agent Loop 学习版
23. LangGraph checkpoint 有状态会话
24. thread_id 多会话隔离
25. LangGraph Agent 结构化返回
26. LangGraph 自定义 State 保存 `tool_calls` 和 `sources`
27. LangGraph 自定义 reducer 实现跨轮状态累积
28. 整理 LangGraph 学习版代码入口
29. FastAPI 新增 `/chat/langgraph` 接口
30. API 层 `session_id` 映射 LangGraph `thread_id`

所以你现在不是零散 demo，而是一个已经初步项目化的 RAG Agent。

---

## 2. 项目整体架构

```text
用户 / curl / Swagger / 命令行
        ↓
FastAPI /chat、/chat/langgraph 或 RAG_Agent_demo.py
        ↓
run_agent(user_query) 或 run_graph_agent(user_query, thread_id=session_id)
        ↓
LLM 判断是否需要调用工具
        ↓
tool_calls
        ↓
Python 执行真实工具
  ├── search_docs：检索本地知识库
  ├── calculator：数学计算
  └── get_weather：模拟天气
        ↓
工具结果写回 messages
        ↓
再次调用 LLM
        ↓
生成最终回答
        ↓
返回结构化 JSON
        ↓
写入 logs/agent.log
```

你可以把整个项目理解成三层：

```text
底层：RAG 知识库和工具函数
中层：Agent Loop 负责决策和调用工具
上层：FastAPI / CLI / Eval 调用 Agent
```

---

## 3. 一次请求是怎么走的

以这个问题为例：

```text
请介绍一下 RAG，并帮我计算 12 加 8
```

完整流程：

```text
1. 用户通过 /chat 或命令行输入问题
2. 程序调用 run_agent(user_query)
3. run_agent 构造 messages
4. 第一次调用 LLM
5. LLM 判断需要两个工具：
   - search_docs
   - calculator
6. 程序读取 message.tool_calls
7. 程序执行 search_docs(query=...)
8. 程序执行 calculator(operation="add", a=12, b=8)
9. 工具结果被包装成 tool message 追加到 messages
10. 第二次调用 LLM
11. LLM 根据知识库结果和计算结果生成最终回答
12. run_agent 返回 dict
13. FastAPI 把 dict 作为 JSON 返回
14. 同时写入 logs/agent.log
```

---

## 4. 核心概念复习

### 4.1 RAG 是什么

RAG 全称：

```text
Retrieval-Augmented Generation
```

中文：

```text
检索增强生成
```

普通 LLM：

```text
用户问题 → LLM → 答案
```

RAG：

```text
用户问题 → 检索知识库 → LLM 基于检索结果回答
```

RAG 的作用：

1. 让模型能回答本地知识库内容
2. 降低幻觉
3. 提供答案来源
4. 不需要重新训练模型

---

### 4.2 Agent 是什么

你现在可以这样理解：

```text
Agent = LLM + Tools + Loop + State
```

其中：

- LLM：负责判断和生成
- Tools：外部能力，例如搜索、计算、查天气
- Loop：模型调用工具后，还要再调用模型
- State：保存上下文，例如 messages

---

### 4.3 Function Calling 是什么

Function Calling / Tool Calling 的核心是：

```text
模型不直接执行 Python 函数；
模型只输出“我想调用哪个工具、参数是什么”；
真正执行工具的是你的 Python 程序。
```

例如模型输出：

```python
{
    "name": "calculator",
    "args": {
        "operation": "multiply",
        "a": 23,
        "b": 19
    }
}
```

程序再执行：

```python
calculator(operation="multiply", a=23, b=19)
```

---

### 4.4 messages 是什么

`messages` 是 Agent 的上下文记忆。

里面会保存：

```text
system：系统提示词
user：用户问题
assistant：模型回复或工具调用请求
tool：工具执行结果
```

Agent 每一轮都依赖 messages 判断下一步该做什么。

---

### 4.5 tool_calls 是什么

`message.tool_calls` 是模型生成的工具调用请求。

它通常包含：

```python
{
    "name": "工具名",
    "args": "工具参数",
    "id": "工具调用 ID"
}
```

程序根据它决定：

```text
要调用哪个工具
传什么参数
工具结果要对应哪个 tool_call_id
```

---

### 4.6 为什么工具结果要写回 messages

因为模型本身不知道 Python 工具执行结果。

所以程序执行完工具后，必须把结果追加回 messages：

```python
messages.append({
    "role": "tool",
    "tool_call_id": tool_call_id,
    "content": json.dumps(function_response, ensure_ascii=False),
})
```

然后再次调用模型，模型才能基于工具结果回答。

---

## 5. 文件复习地图

### 5.1 `config.py`

作用：项目配置。

你需要记住：

```python
DOCS_DIR = "./docs"
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "rag_demo_collection"
MAX_AGENT_ROUNDS = 5
```

重点：

- `.env` 在这里加载
- `MAX_AGENT_ROUNDS` 防止 Agent 无限循环

---

### 5.2 `models.py`

作用：初始化模型。

包含两个核心对象：

```python
llm
embeddings
```

`llm`：

- 用于聊天
- 用于判断工具调用
- 用于生成最终回答

`embeddings`：

- 用于把文档和用户问题转成向量
- 用于 Chroma 语义检索

当前 Embedding：

```text
BAAI/bge-m3
```

---

### 5.3 `vector_store.py`

作用：构建或加载本地向量库。

流程：

```text
load_local_docs()
  ↓
split_docs()
  ↓
load_or_create_vector_store()
```

你需要掌握：

#### `load_local_docs`

读取 `docs/` 下的 `.md` / `.txt` 文件，变成 LangChain `Document`。

#### `split_docs`

把长文档切成小片段。

当前参数：

```python
chunk_size=100
chunk_overlap=20
```

#### `load_or_create_vector_store`

如果 Chroma 已存在且 docs 没变，就直接加载。  
如果 docs 变了，就重建向量库。

---

### 5.4 `tools.py`

作用：定义真实 Python 工具。

当前有三个工具：

#### `search_docs`

调用 Chroma：

```python
vector_store.similarity_search(query, k=k)
```

返回：

```python
[
    {
        "source": "xxx.md",
        "content": "检索片段"
    }
]
```

#### `calculator`

支持：

```text
add / subtract / multiply / divide
```

#### `get_weather`

模拟天气数据，不是真实 API。

---

### 5.5 `schemas.py`

作用：定义给模型看的工具说明书。

注意：

```text
tools.py 是给 Python 程序执行的
schemas.py 是给模型看的
```

模型通过 schema 知道：

1. 有哪些工具
2. 每个工具干什么
3. 每个工具需要什么参数
4. 参数类型是什么

---

### 5.6 `agent.py`

作用：项目核心，手写 Agent Loop。

核心函数：

```python
run_agent(user_query: str) -> dict
```

核心流程：

```text
构造 messages
  ↓
调用 llm.invoke(messages, tools=TOOLS, tool_choice="auto")
  ↓
如果没有 tool_calls：返回最终答案
  ↓
如果有 tool_calls：执行工具
  ↓
工具结果追加到 messages
  ↓
再次调用模型
```

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

你在这个文件里完成的工程化能力：

1. 结构化返回
2. 工具调用记录
3. RAG 来源追踪
4. 工具异常保护
5. 最大轮数限制
6. JSONL 日志

---

### 5.7 `RAG_Agent_demo.py`

作用：命令行入口。

它做的事：

```text
打印项目初始化信息
循环读取用户输入
调用 run_agent(query)
打印结构化返回
```

---

### 5.8 `app/main.py`

作用：FastAPI 服务入口。

当前接口：

```text
GET /health
POST /chat
POST /chat/langgraph
```

`/chat` 流程：

```text
接收 JSON：{"message": "..."}
  ↓
校验 message 是否为空
  ↓
调用 run_agent()
  ↓
返回结构化 JSON
```

目前已经有：

- 空输入校验
- try/except 捕获 Agent 异常
- HTTP 400
- HTTP 500

`/chat/langgraph` 流程：

```text
接收 JSON：{"message": "...", "session_id": "..."}
  ↓
校验 message 和 session_id 是否为空
  ↓
调用 run_graph_agent(user_query=message, thread_id=session_id)
  ↓
返回 LangGraph Agent 的结构化 JSON
```

其中：

```text
session_id 是 API 层会话 ID
thread_id 是 LangGraph checkpointer 使用的会话 ID
```

项目中做了映射：

```text
session_id → thread_id
```

所以同一个 `session_id` 可以保留上下文，不同 `session_id` 之间状态隔离。

---

### 5.9 `eval/questions.json`

作用：评估问题集。

每条数据包含：

```json
{
  "id": "calc_001",
  "question": "帮我计算 23 乘以 19",
  "expected_tools": ["calculator"]
}
```

---

### 5.10 `eval/run_eval.py`

作用：最小 Agent 评估脚本。

评估流程：

```text
读取 questions.json
  ↓
调用 run_agent()
  ↓
提取实际调用工具 actual_tools
  ↓
和 expected_tools 对比
  ↓
统计通过率
```

当前评估的是：

```text
工具调用是否正确
```

还没有评估：

- 工具参数是否完全正确
- RAG 检索来源是否正确
- 最终答案是否正确

---

### 5.11 `LangGraph_learning/step2_agent_loop_graph.py`

作用：LangGraph 学习版 Agent Loop。

它把手写 Agent Loop 映射成图：

```text
START → model → 条件判断
              ├── tools → model
              └── END
```

你要记住这个映射：

| 手写 Agent Loop | LangGraph |
|---|---|
| `messages` | `State` |
| 调用模型 | `model node` |
| 执行工具 | `tools node` |
| `if message.tool_calls` | `conditional edge` |
| while / for 循环 | graph loop |

当前这个文件还新增了：

1. `InMemorySaver` checkpoint
2. `thread_id` 会话 ID
3. 同一个 thread_id 下的历史记忆
4. 不同 thread_id 之间的会话隔离
5. `run_graph_agent()` 结构化返回 dict
6. 自定义 `GraphState`
7. `tool_calls` / `sources` 状态字段
8. `merge_tool_calls` / `merge_sources` 自定义 reducer
9. `main_chat_loop()` 命令行入口
10. 测试函数和交互入口分离

当前 `run_graph_agent()` 返回：

```python
{
    "thread_id": "user-a",
    "answer": "你叫小明。",
    "tool_calls": [...],
    "sources": [...],
    "messages_count": 4
}
```

其中：

- `thread_id`：当前会话 ID
- `answer`：模型最终回答
- `tool_calls`：当前会话累计工具调用记录
- `sources`：当前会话累计 RAG 来源，已去重
- `messages_count`：当前会话累计消息数量

---

## 6. 三条主线复习

### 6.1 RAG 主线

```text
docs 文件
  ↓
load_local_docs
  ↓
Document
  ↓
split_docs
  ↓
chunks
  ↓
embedding
  ↓
Chroma
  ↓
search_docs
  ↓
返回 source + content
```

你要能回答：

- 为什么要切分？
- chunk_size 和 chunk_overlap 是什么？
- embedding 是什么？
- Chroma 存什么？
- search_docs 返回什么？

---

### 6.2 Agent 主线

```text
user_query
  ↓
messages
  ↓
llm.invoke(..., tools=TOOLS)
  ↓
message.tool_calls
  ↓
AVAILABLE_FUNCTIONS[function_name](**function_args)
  ↓
Tool result
  ↓
messages.append(tool message)
  ↓
llm.invoke(...)
  ↓
final answer
```

你要能回答：

- tool_calls 是什么？
- schema 和真实工具函数有什么区别？
- 为什么工具结果要返回给模型？
- 为什么需要最大轮数？
- 工具异常怎么处理？

---

### 6.3 工程化主线

```text
run_agent 返回 dict
  ↓
FastAPI /chat 直接返回 JSON
  ↓
logs/agent.log 记录执行过程
  ↓
eval/run_eval.py 自动评估工具调用
```

你要能回答：

- 为什么不能只 print？
- 为什么要记录 tool_calls？
- 为什么要记录 sources？
- 为什么要写日志？
- 为什么要做评估？
- FastAPI 起什么作用？

---

### 6.4 LangGraph 有状态 Agent 主线

```text
StateGraph(MessagesState)
  ↓
添加 model 节点
  ↓
添加 tools 节点
  ↓
添加 conditional edge
  ↓
添加 tools → model 回边
  ↓
InMemorySaver checkpoint
  ↓
invoke 时传 configurable.thread_id
  ↓
同一个 thread_id 复用历史 messages
  ↓
不同 thread_id 状态隔离
```

你要能回答：

- checkpoint 是什么？
- InMemorySaver 保存在哪里？
- thread_id 是什么？
- 为什么同一个 thread_id 能记住历史？
- 为什么不同 thread_id 之间不会串话？
- messages_count 为什么会随着对话增加？

---

### 6.5 LangGraph 自定义 State 与 reducer 主线

你现在的 LangGraph 学习版不再只保存 messages，而是定义了：

```python
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls: Annotated[list[dict], merge_tool_calls]
    sources: Annotated[list[str], merge_sources]
```

含义：

```text
messages：对话历史，用 add_messages 追加合并
tool_calls：工具调用记录，用 merge_tool_calls 累积
sources：RAG 来源，用 merge_sources 累积并去重
```

你要能回答：

- 为什么不能只用 `MessagesState`？
- `Annotated[list, add_messages]` 是什么意思？
- reducer 是什么？
- `merge_tool_calls` 做了什么？
- `merge_sources` 为什么要去重？
- 为什么 `tool_calls` 和 `sources` 不加 reducer 会被覆盖？

---

## 7. 当前运行方式复习

### 7.1 命令行运行 Agent

```bash
python RAG_Agent_demo.py
```

---

### 7.2 启动 FastAPI

```bash
uvicorn app.main:app --reload
```

接口文档：

```text
http://127.0.0.1:8000/docs
```

测试 `/chat`：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "帮我计算 23 乘以 19"}'
```

测试 `/chat/langgraph` 有状态会话：

```bash
curl -X POST http://127.0.0.1:8000/chat/langgraph \
  -H "Content-Type: application/json" \
  -d '{"message": "我叫小明", "session_id": "user-a"}'
```

```bash
curl -X POST http://127.0.0.1:8000/chat/langgraph \
  -H "Content-Type: application/json" \
  -d '{"message": "我叫什么？", "session_id": "user-a"}'
```

---

### 7.3 运行评估

```bash
python eval/run_eval.py
```

---

### 7.4 运行 LangGraph 学习版

```bash
python LangGraph_learning/step2_agent_loop_graph.py
```

---

## 8. 你现在最容易混淆的点

### 8.1 `tools.py` 和 `schemas.py` 的区别

```text
tools.py：真实 Python 函数，给程序执行
schemas.py：工具说明书，给模型看
```

模型不会直接执行 `tools.py`。  
模型只根据 `schemas.py` 输出工具调用请求。  
程序再根据工具名去 `AVAILABLE_FUNCTIONS` 里找到真实函数执行。

---

### 8.2 RAG 和 Agent 的区别

RAG：

```text
检索知识库 → 回答
```

Agent：

```text
模型决定是否调用工具 → 工具执行 → 模型回答
```

你的项目是：

```text
把 RAG 检索封装成 Agent 的一个工具
```

---

### 8.3 `message.tool_calls` 和工具结果的区别

`message.tool_calls`：

```text
模型说：“我想调用这个工具，参数是这些。”
```

工具结果：

```text
Python 程序真的执行工具后得到的结果。
```

二者不是一回事。

---

### 8.4 `tool_call_id` 为什么重要

模型可能一次调用多个工具。

`tool_call_id` 用来对应：

```text
哪个工具请求 → 哪个工具结果
```

所以返回工具结果时必须带上它。

---

### 8.5 为什么要用 LangGraph

简单 Agent 手写 Loop 可以。

复杂 Agent 用 LangGraph 更清晰：

```text
State
Node
Edge
Conditional Edge
```

它把隐式 if/else/while 变成显式图结构。

---

### 8.6 checkpoint 和 thread_id 的区别

`checkpoint`：

```text
负责保存状态。
```

当前保存的主要状态是：

```python
{
    "messages": [...]
}
```

`thread_id`：

```text
负责区分状态属于哪个会话。
```

例如：

```text
thread_id="user-a"
thread_id="user-b"
```

同一个 `thread_id` 会延续历史消息。  
不同 `thread_id` 的历史消息互相隔离。

---

### 8.7 InMemorySaver 的特点

你当前用的是：

```python
InMemorySaver()
```

它的特点：

```text
状态保存在当前 Python 进程的内存中。
```

优点：

- 简单
- 适合学习
- 适合 demo

缺点：

- 程序关闭后状态消失
- 不适合生产环境长期保存

生产环境可以换成：

- SQLite checkpointer
- Postgres checkpointer
- Redis / 其他持久化存储

---

### 8.8 messages_count 为什么第二轮增加 2

`messages_count` 不是对话轮数，而是当前会话累计消息条数。

普通一轮对话通常会增加两条消息：

```text
user message
assistant message
```

如果调用工具，一轮会增加更多：

```text
user
assistant(tool_call)
tool
assistant(final)
```

所以调用一次工具通常至少增加 4 条消息。

---

### 8.9 add_messages 和自定义 reducer

LangGraph 的 State 字段可以定义“合并规则”。

例如：

```python
messages: Annotated[list, add_messages]
```

表示：

```text
节点返回新的 messages 时，不覆盖旧 messages，而是追加到旧 messages 后面。
```

你又自定义了：

```python
tool_calls: Annotated[list[dict], merge_tool_calls]
sources: Annotated[list[str], merge_sources]
```

含义：

```text
tool_calls：旧工具调用 + 新工具调用
sources：旧来源 + 新来源，并去重
```

如果不加 reducer，节点返回新值时可能覆盖旧值。

---

### 8.10 LangGraph 学习版代码结构

现在 `step2_agent_loop_graph.py` 底部结构已经整理成：

```python
def test_thread_isolation() -> None:
    ...

def test_tool_state_accumulation() -> None:
    ...

def main_chat_loop() -> None:
    ...

if __name__ == "__main__":
    main_chat_loop()
```

职责：

- `run_graph_agent()`：核心运行函数
- `test_thread_isolation()`：测试不同 thread_id 的会话隔离
- `test_tool_state_accumulation()`：测试 tool_calls / sources 跨轮累积
- `main_chat_loop()`：正常命令行聊天入口
- `__main__`：默认进入聊天模式

---

## 9. 当前项目亮点

你面试时可以重点说：

1. 我不是只做了普通 RAG，而是把 RAG 封装成 Agent 工具
2. Agent 能根据问题自动选择工具
3. 支持多工具调用
4. 有工具异常处理
5. 有最大轮数限制，防止死循环
6. 返回结构化 JSON，不只是 print
7. 记录 tool_calls，方便调试和评估
8. 记录 sources，方便答案溯源
9. 有 JSONL 日志
10. 有最小评估脚本
11. 用 FastAPI 封装成 HTTP 服务
12. 学习了 LangGraph 图结构表达 Agent Loop
13. 使用 LangGraph checkpoint 实现有状态对话
14. 使用 thread_id 验证多会话隔离
15. LangGraph Agent 支持结构化返回
16. 使用自定义 reducer 累积 tool_calls 和 sources
17. 将测试函数和交互入口分离，代码结构更清晰
18. 将 LangGraph Agent 接入 FastAPI，提供 `/chat/langgraph`
19. 通过 `session_id` 实现 API 层有状态会话

---

## 10. 当前不足

当前项目还不是生产级系统，主要不足：

1. RAG 没有 reranker
2. chunk_size 还没有系统调优
3. 评估只评估工具调用，没有评估答案质量
4. 天气工具是模拟数据
5. 还没有 Docker
6. 还没有前端页面
7. LangGraph 版本还只是学习版，没有接入 FastAPI 主流程
8. 没有多用户 session / thread_id
9. 没有持久化 checkpoint
10. 没有 vLLM 本地模型部署

---

## 11. 下一步建议

如果你现在想复习，不要急着继续加新功能。

建议顺序：

1. 重新读 `agent.py`
2. 重新读 `tools.py`
3. 重新读 `schemas.py`
4. 重新读 `vector_store.py`
5. 运行 `python RAG_Agent_demo.py`
6. 运行 `python eval/run_eval.py`
7. 启动 `uvicorn app.main:app --reload`
8. 用 Swagger 调 `/chat`
9. 再读 `LangGraph_learning/step2_agent_loop_graph.py`

确认你能用自己的话讲清楚：

```text
RAG 怎么做
Agent Loop 怎么跑
工具怎么调用
FastAPI 怎么封装
评估脚本怎么评估
LangGraph 和手写 Loop 的对应关系
```

---

## 12. 给你自己的复习问题

你可以用下面这些问题自测。

### RAG

1. `docs/` 里的文件是怎么进入 Chroma 的？
2. 为什么要切分文档？
3. Embedding 的作用是什么？
4. `search_docs` 返回什么？
5. `sources` 是怎么来的？

### Agent

1. `messages` 里保存了什么？
2. `message.tool_calls` 是什么？
3. schema 和工具函数有什么区别？
4. 工具结果为什么要追加回 messages？
5. 为什么需要 `MAX_AGENT_ROUNDS`？

### 工程化

1. 为什么 `run_agent()` 要返回 dict？
2. `tool_call_records` 有什么用？
3. `logs/agent.log` 记录了什么？
4. `eval/run_eval.py` 怎么判断通过？
5. FastAPI `/chat` 做了什么？

### LangGraph

1. State 对应手写 Loop 里的什么？
2. Node 对应什么？
3. Conditional Edge 对应什么？
4. 为什么 tools 节点执行完要回到 model 节点？
5. checkpoint 的作用是什么？
6. thread_id 的作用是什么？
7. InMemorySaver 有什么特点？
8. 如何验证不同 thread_id 的会话隔离？
9. messages_count 为什么不等于对话轮数？
10. reducer 的作用是什么？
11. add_messages 和 merge_tool_calls 有什么区别？
12. 为什么 sources 要去重？

---

## 13. 一段面试版项目介绍

你可以这样介绍：

> 我做了一个基于 RAG 和 Function Calling 的多工具 Agent 项目。底层用本地 docs 文档构建 Chroma 向量库，使用 bge-m3 做 embedding，并把知识库检索封装成 `search_docs` 工具。Agent 层通过手写 Agent Loop 调用模型，让模型根据用户问题自动选择 `search_docs`、`calculator` 或 `get_weather`。工具执行后，结果会作为 tool message 返回给模型，模型再生成最终回答。工程化方面，我实现了结构化返回、工具调用记录、RAG 来源追踪、异常处理、最大轮数限制、JSONL 日志、最小工具调用评估脚本，并用 FastAPI 封装了 `/chat` 接口。后续我学习了用 LangGraph 把手写 Agent Loop 改造成显式状态图，并通过 checkpoint 和 thread_id 实现有状态对话和多会话隔离，为后续扩展复杂工作流打基础。

现在也可以补充：

> 我进一步把 LangGraph Agent 接入 FastAPI，新增 `/chat/langgraph` 接口。API 请求里的 `session_id` 会映射到 LangGraph 的 `thread_id`，从而支持同一会话保留上下文、不同会话状态隔离，以及工具调用记录和 RAG 来源的跨轮累积。
