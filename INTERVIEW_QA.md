# RAG Agent 项目面试问题整理

> 这份文档用于面试复习。重点不是背答案，而是能结合你自己的项目讲清楚。

---

## 1. 项目介绍类

### Q1：请介绍一下你的项目。

答：

我做的是一个基于 RAG 和 Function Calling 的多工具 Agent 项目。系统支持本地知识库问答、数学计算和天气查询。用户输入问题后，模型会根据问题自动判断是否需要调用工具，例如知识库问题调用 `search_docs`，数学问题调用 `calculator`，天气问题调用 `get_weather`。工具执行完成后，结果会返回给模型，由模型生成最终回答。

项目还做了工程化增强，包括结构化返回、工具调用记录、RAG 来源追踪、异常处理、日志记录、最小评估脚本，以及 FastAPI 服务化接口。

---

### Q2：这个项目和普通 RAG demo 有什么区别？

答：

普通 RAG demo 通常是固定流程：

```text
用户问题 → 检索 → 回答
```

我的项目把 RAG 检索封装成了 Agent 的一个工具 `search_docs`。也就是说，系统不是每次都强制检索，而是让模型根据问题判断是否需要检索。

例如：

- 问知识库内容时调用 `search_docs`
- 问计算题时调用 `calculator`
- 问普通问题时可以直接回答
- 混合问题可以同时调用多个工具

此外，我还加入了日志、评估和 FastAPI 服务化，不只是命令行 demo。

---

### Q3：你项目目前有哪些功能？

答：

目前功能包括：

1. 本地 `.md` / `.txt` 文档加载
2. 文档切分
3. bge-m3 embedding
4. Chroma 向量库
5. 知识库检索工具
6. 计算器工具
7. 模拟天气工具
8. Function Calling Agent Loop
9. 工具异常处理
10. RAG 来源追踪
11. JSONL 日志记录
12. 最小工具调用评估
13. FastAPI `/chat` 接口
14. Swagger 文档

---

## 2. RAG 基础类

### Q4：RAG 的完整流程是什么？

答：

RAG 的流程是：

```text
文档加载 → 文档切分 → Embedding → 存入向量库 → 用户提问 → 检索相关片段 → 拼接上下文 → LLM 生成回答
```

在我的项目中，文档放在 `docs/` 目录，通过 `vector_store.py` 加载和切分，使用 `BAAI/bge-m3` 做 embedding，用 Chroma 持久化存储。用户问题通过 `search_docs` 工具检索相关文档片段，再交给模型回答。

---

### Q5：为什么要做文档切分？

答：

主要有三个原因：

1. 长文档无法全部放进 prompt
2. 向量检索通常以片段为单位
3. 小片段更容易匹配用户具体问题

但切分太小会丢上下文，太大又会引入噪声。所以需要根据文档类型调整 `chunk_size` 和 `chunk_overlap`。

---

### Q6：chunk_size 和 chunk_overlap 怎么理解？

答：

`chunk_size` 是每个文档片段的大致长度。  
`chunk_overlap` 是相邻片段之间重叠的内容长度。

overlap 的作用是避免重要语义刚好被切断。

例如一句话跨越两个 chunk，如果没有 overlap，检索时可能只召回半句话，影响回答质量。

---

### Q7：Embedding 是什么？

答：

Embedding 是把文本转换成向量表示。

语义相近的文本，向量距离也更近。RAG 就是利用这个特性，把用户问题转成向量，然后在向量数据库里找语义最相近的文档片段。

我的项目使用 `BAAI/bge-m3`，适合中英文检索。

---

### Q8：为什么用 Chroma？

答：

Chroma 是一个轻量级向量数据库，适合本地开发和 demo。

它可以持久化保存文档向量，并支持相似度搜索。对于当前项目规模，Chroma 简单、易用、部署成本低。

如果项目进入生产，可以考虑 Milvus、Qdrant、Elasticsearch hybrid search 等方案。

---

### Q9：RAG 如何减少幻觉？

答：

RAG 通过外部知识库给模型提供依据，让模型基于检索到的内容回答，而不是完全依赖模型参数记忆。

我的项目还做了两点：

1. `search_docs` 返回 `source`
2. Prompt 要求如果使用知识库结果，需要在答案末尾列出参考来源；如果没有相关信息，要说明知识库中没有找到，不要编造

---

## 3. Agent / Function Calling 类

### Q10：什么是 Agent？

答：

Agent 可以理解为：

```text
LLM + Tools + Loop/Workflow + State
```

它不只是回答问题，还能根据任务决定是否调用工具，拿到工具结果后继续推理或生成最终答案。

---

### Q11：Function Calling 的原理是什么？

答：

我们先把工具 schema 传给模型，告诉模型有哪些工具、工具参数是什么。模型不会直接执行函数，而是生成结构化的工具调用请求，比如调用 `calculator` 并传入参数。

程序收到 `tool_calls` 后，根据工具名找到对应 Python 函数并执行。执行结果再作为 tool message 返回给模型，模型基于结果生成最终回答。

---

### Q12：为什么需要 Agent Loop？

答：

因为模型可能需要多轮工具调用。

流程是：

```text
调用模型 → 如果有 tool_calls 就执行工具 → 工具结果写回 messages → 再调用模型
```

直到模型不再调用工具，说明它可以生成最终答案。

---

### Q13：如何防止 Agent 无限循环？

答：

设置最大循环轮数。

我的项目里配置了：

```python
MAX_AGENT_ROUNDS = 5
```

如果超过最大轮数，系统会停止继续调用工具，并返回兜底信息。这是 Agent 稳定性的基本措施。

---

### Q14：工具调用失败怎么办？

答：

不能让工具异常导致整个 Agent 崩溃。

我的做法是：

```python
try:
    function_response = tool(**args)
except Exception as e:
    function_response = {"error": str(e)}
```

然后把错误作为工具结果返回给模型，让模型生成友好的错误说明。

---

### Q15：为什么要记录 tool_calls？

答：

因为 Agent 的关键不只是最终答案，还包括中间决策过程。

记录 tool_calls 可以看到：

1. 调用了哪个工具
2. 传了什么参数
3. 工具返回了什么结果

这些信息对调试、日志、评估和面试展示都很重要。

---

## 4. 工程化类

### Q16：为什么 `run_agent()` 要返回 dict，而不是只 print？

答：

因为项目级 Agent 需要被其他模块调用，比如 FastAPI、评估脚本和前端。

如果只 print，外部拿不到结果。返回 dict 后，可以直接作为 JSON 返回，也可以写日志、做评估。

---

### Q17：你的结构化返回里有哪些字段？

答：

当前返回：

```python
{
    "user_query": "用户问题",
    "answer": "最终回答",
    "tool_calls": [...],
    "sources": [...],
    "rounds": 2
}
```

其中：

- `answer` 是最终回答
- `tool_calls` 是工具调用记录
- `sources` 是 RAG 来源
- `rounds` 是 Agent 运行轮数

---

### Q18：为什么要写日志？

答：

Agent 行为具有不确定性，所以需要记录运行过程。

我的项目把每次运行结果写入 `logs/agent.log`，每行一条 JSON，包含用户问题、最终答案、工具调用、来源和轮数。

这样方便后续分析和调试。

---

### Q19：FastAPI 在项目中起什么作用？

答：

FastAPI 用来把 Agent 封装成 HTTP 服务。

当前有两个接口：

```text
GET /health
POST /chat
```

这样前端、curl、Postman 或其他系统都可以通过 HTTP 调用 Agent，而不是只能在命令行里运行。

---

### Q20：Pydantic 起什么作用？

答：

Pydantic 用来定义请求体和做参数校验。

例如：

```python
class ChatRequest(BaseModel):
    message: str
```

FastAPI 会根据这个模型自动解析 JSON、校验字段，并生成 Swagger 文档。

---

## 5. 评估类

### Q21：你是怎么评估 Agent 的？

答：

我目前做了一个最小评估脚本，重点评估工具调用是否正确。

流程是：

1. 在 `eval/questions.json` 中维护问题和期望工具
2. `eval/run_eval.py` 逐条调用 `run_agent`
3. 从返回结果中提取实际调用工具
4. 判断期望工具是否都被调用
5. 统计通过率

当前测试覆盖计算、天气、RAG 和混合问题。

---

### Q22：为什么先评估工具调用，而不是答案质量？

答：

因为 Agent 的第一步是决策是否正确。如果工具都没有调对，最终答案质量就没有保障。

所以我先做最小可行评估：检查 expected_tools 是否出现在 actual_tools 中。

后续可以继续扩展到：

- 参数是否正确
- RAG 来源是否命中
- 答案是否包含关键事实
- 是否幻觉

---

## 6. 项目不足与优化类

### Q23：你这个项目目前有什么不足？

答：

目前不足包括：

1. RAG 没有 reranker
2. 评估只覆盖工具调用，没有覆盖答案质量
3. 天气工具是模拟数据
4. FastAPI 还需要加强错误处理和请求校验
5. 还没有 Docker 部署
6. 还没有前端
7. 还没有 LangGraph 版本的状态图编排

---

### Q24：如果继续优化，你会怎么做？

答：

我会按优先级做：

1. 完善 FastAPI 错误处理和请求校验
2. 增加 RAG 答案质量评估
3. 加入 reranker 提升检索质量
4. 增加真实外部 API 工具
5. Docker 化部署
6. 使用 LangGraph 重构 Agent Loop，实现更清晰的状态流转和 checkpoint
7. 接入 vLLM 部署本地 Qwen 模型

---

### Q25：LangGraph 和你现在手写 Agent Loop 有什么关系？

答：

我现在手写的 Agent Loop 是理解 Agent 的基础。

LangGraph 可以把这个流程图结构化：

```text
messages/state → model node → tools node → conditional edge → model node
```

也就是说：

- `messages` 对应 State
- 调用模型对应 model node
- 执行工具对应 tools node
- 判断是否继续对应 conditional edge

先手写 Agent Loop 能帮助我理解 LangGraph 的底层逻辑。

---

### Q25-1：LangGraph 的 checkpoint 有什么作用？

答：

checkpoint 用来保存图运行过程中的 state。当前我的 LangGraph 学习版使用的是 `MessagesState`，所以主要保存的是 `messages`。

没有 checkpoint 时，每次调用只看当前输入；加了 checkpoint 后，同一个 `thread_id` 下可以延续之前的对话历史，实现有状态多轮对话。

---

### Q25-2：thread_id 是什么？

答：

`thread_id` 可以理解为会话 ID 或用户 ID。LangGraph 通过 `configurable.thread_id` 判断当前调用属于哪个会话。

同一个 `thread_id` 会复用历史状态，不同 `thread_id` 的状态相互隔离。

---

### Q25-3：你怎么验证 LangGraph 的会话隔离？

答：

我做了一个测试：

```python
run_graph_agent("我叫小明", thread_id="user-a")
run_graph_agent("我叫什么？", thread_id="user-b")
run_graph_agent("我叫什么？", thread_id="user-a")
```

结果是：

```text
user-b 不知道“小明”
user-a 能回答“小明”
```

这说明不同 `thread_id` 的状态是隔离的，同一个 `thread_id` 能延续历史。

---

### Q25-4：InMemorySaver 有什么特点？

答：

`InMemorySaver` 是 LangGraph 的内存版 checkpointer。

优点是简单，适合学习和 demo；缺点是状态只保存在当前 Python 进程中，程序关闭后就会丢失。

生产环境可以换成 SQLite、Postgres 等持久化 checkpointer。

---

### Q25-5：messages_count 为什么第二轮会增加 2？

答：

因为 `messages_count` 统计的是当前会话累计消息条数，不是对话轮数。

普通一轮对话会新增：

```text
user message
assistant message
```

所以同一个 `thread_id` 下第二轮普通对话会比第一轮多 2 条消息。

如果这一轮调用工具，则会新增：

```text
user
assistant(tool_call)
tool
assistant(final)
```

通常至少增加 4 条消息。

---

### Q25-6：LangGraph 里的 reducer 是什么？

答：

reducer 是 LangGraph 用来合并 State 字段的函数。节点返回新状态时，LangGraph 需要知道新值应该覆盖旧值，还是和旧值合并。

比如：

```python
messages: Annotated[list, add_messages]
```

表示 messages 使用 `add_messages` 合并，也就是追加新消息。

我还自定义了 `merge_tool_calls` 和 `merge_sources`，让工具调用记录和 RAG 来源也能跨多轮累积。

---

### Q25-7：为什么要自定义 GraphState？

答：

一开始用 `MessagesState` 只能保存 messages。后来我希望 LangGraph Agent 也能像主项目 `agent.py` 一样返回结构化信息，比如工具调用记录和 RAG 来源，所以我自定义了 `GraphState`：

```python
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls: Annotated[list[dict], merge_tool_calls]
    sources: Annotated[list[str], merge_sources]
```

这样图状态中不仅有对话消息，还有业务字段。

---

### Q25-8：merge_tool_calls 和 merge_sources 分别做什么？

答：

`merge_tool_calls` 用来累积工具调用记录：

```text
旧 tool_calls + 新 tool_calls
```

`merge_sources` 用来累积 RAG 来源，并做去重：

```text
旧 sources + 新 sources，再去重
```

这样同一个 `thread_id` 下，多轮调用工具时，历史工具调用和来源不会丢失。

---

### Q25-9：你怎么验证 tool_calls 和 sources 可以跨轮累积？

答：

我写了 `test_tool_state_accumulation()`，用同一个 `thread_id` 连续问两个问题：

```text
第一轮：帮我计算 23 乘以 19
第二轮：RAG 是什么？
```

如果 reducer 生效，第二轮返回的 `tool_calls` 应该同时包含：

```text
calculator
search_docs
```

测试结果符合预期，说明 `tool_calls` 可以跨轮累积，`sources` 也能保留 RAG 来源。

---

### Q25-10：你是怎么把 LangGraph Agent 接入 FastAPI 的？

答：

我新增了一个接口：

```text
POST /chat/langgraph
```

请求体包含：

```json
{
  "message": "用户问题",
  "session_id": "user-a"
}
```

接口内部调用：

```python
run_graph_agent(
    user_query=request.message,
    thread_id=request.session_id,
)
```

也就是说，API 层的 `session_id` 会映射为 LangGraph 的 `thread_id`，从而实现有状态多轮对话。

---

### Q25-11：`/chat` 和 `/chat/langgraph` 有什么区别？

答：

`/chat` 调用的是我手写的 Agent Loop，适合理解 Function Calling 的底层流程，比如 messages、tool_calls、工具执行和 tool message。

`/chat/langgraph` 调用的是 LangGraph 版本，它把流程显式建模成 StateGraph，并支持 checkpoint 和 thread_id，因此可以实现同一 session_id 保留历史、不同 session_id 状态隔离。

---

### Q25-12：为什么 API 层叫 session_id，而 LangGraph 里叫 thread_id？

答：

`session_id` 是面向 API 使用者的概念，更容易理解为一次用户会话。

`thread_id` 是 LangGraph checkpointer 用来区分不同对话线程的配置项。

所以我在 FastAPI 层做了映射：

```text
session_id → thread_id
```

这样既符合 API 语义，也能使用 LangGraph 的状态管理能力。

---

## 7. 高频追问

### Q26：模型不调用工具怎么办？

答：

可以从几个方面排查：

1. 工具 schema 的 description 是否清楚
2. system prompt 是否明确要求何时调用工具
3. 用户问题是否足够明确
4. 模型是否支持 tool calling
5. 是否设置了 `tool_choice="auto"`

---

### Q27：模型乱调用工具怎么办？

答：

可以：

1. 优化 system prompt，明确工具使用边界
2. 改进工具 description
3. 增加 router 或规则判断
4. 在工具层做参数校验
5. 用评估集回归测试工具调用准确率

---

### Q28：RAG 检索不到怎么办？

答：

排查方向：

1. 文档是否正确加载
2. 文档切分是否合理
3. embedding 模型是否适合
4. query 是否需要改写
5. top-k 是否太小
6. 是否需要 rerank
7. 文档源中是否真的有答案

---

### Q29：为什么答案要带来源？

答：

来源可以让用户验证答案依据，降低幻觉风险。企业知识库场景中，可追溯性很重要。

---

### Q30：这个项目如何体现你的工程能力？

答：

它不只是一个能跑的 demo，而是包含了工程化要素：

1. 模块化代码结构
2. 配置管理
3. 向量库持久化
4. Agent 工具调用
5. 错误处理
6. 结构化返回
7. 日志记录
8. 自动评估
9. FastAPI 服务化
