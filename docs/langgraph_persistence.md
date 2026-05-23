# LangGraph Persistence：Checkpoint、Thread ID 与有状态 Agent

> 本文是学习型知识库文档，基于 LangGraph 官方 Persistence / Memory 文档整理，用于本项目 RAG 检索与 chunk-level 评估实验。本文不是官方原文复制，而是面向项目实践的中文总结。

## 1. 为什么 Agent 需要持久化

在普通的大模型调用中，模型本身不会自动保存历史。每一次调用模型时，程序都需要把历史消息、系统提示词、工具结果等上下文重新传给模型。如果程序没有保存这些状态，下一轮对话就无法知道上一轮发生了什么。

LangGraph 的 persistence 机制就是为了解决这个问题。它可以在图执行过程中保存状态快照，让一个 Agent 工作流在多轮调用之间保留上下文，也可以在中断、人工审批、调试回放等场景中恢复执行。

在企业知识库 Agent 中，持久化的价值主要体现在三个方面：

1. **多轮对话记忆**：同一个用户可以连续追问，例如先问“RAG 是什么”，再问“它和微调有什么区别”。第二个问题依赖第一轮上下文。
2. **多会话隔离**：不同用户或不同浏览器会话应该拥有各自的历史，不能互相串话。
3. **可调试和可追踪**：开发者可以查看某个会话在某一步的 state，定位工具调用、检索结果或模型回答的问题。

## 2. Checkpoint 是什么

Checkpoint 可以理解为“图状态的快照”。LangGraph 的图由多个节点组成，例如 `call_model`、`call_tools`、`route_tools`。每个节点执行后，图的 state 可能发生变化：messages 变多了、tool_calls 新增了、sources 新增了。Checkpoint 会把某个时间点的 state 保存下来。

在 Agent 场景中，checkpoint 保存的通常不是模型参数，而是运行时状态，例如：

- 当前会话的 messages；
- 用户输入和模型回复；
- AIMessage 中的 tool_calls；
- ToolMessage 中的工具执行结果；
- 自定义 state 字段，例如 sources、rounds、tool_call_records；
- 图下一步要执行的节点信息。

因此，checkpoint 更像是“工作流执行现场”的保存，而不是模型训练里的参数 checkpoint。

## 3. Thread ID 是什么

Thread ID 是 LangGraph 用来区分不同会话的标识。可以把它理解成会话 ID。使用 checkpointer 编译图之后，调用图时通常需要在 config 里传入：

```python
config = {"configurable": {"thread_id": "user-a"}}
```

LangGraph 会根据 `thread_id` 找到对应的历史 checkpoint，然后在旧状态基础上继续执行。如果下一次仍然用 `user-a` 调用图，就能读取到 user-a 的历史 messages。如果换成 `user-b`，LangGraph 会使用另一条独立状态链，因此 user-b 不会看到 user-a 的历史。

在 FastAPI 项目中，通常会把接口层的 `session_id` 映射为 LangGraph 的 `thread_id`：

```text
HTTP request session_id  ->  LangGraph thread_id
```

这样，前端只需要传 session_id，后端就可以实现多轮对话和多会话隔离。

## 4. InMemorySaver 的作用和限制

`InMemorySaver` 是 LangGraph 提供的内存级 checkpointer。它把 checkpoint 保存在当前 Python 进程的内存中。它适合学习、Demo、单进程开发和本地调试。

它的优点是使用简单，不需要数据库：

```python
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
```

但它也有明显限制：

1. Python 进程关闭后，内存数据消失，会话历史也消失。
2. 多进程部署时，不同进程之间不能共享内存 checkpoint。
3. 不适合生产环境长期保存用户会话。
4. 服务重启后无法恢复历史状态。

因此，在生产环境中更常见的做法是使用数据库型 checkpoint，例如 PostgreSQL、Redis 或其他持久化存储。

## 5. Checkpoint 和 messages 的关系

在 LangGraph Agent 中，messages 通常是 state 中最核心的字段。一次完整工具调用可能包含：

1. HumanMessage：用户问题；
2. AIMessage：模型决定调用工具，并带有 tool_calls；
3. ToolMessage：程序执行工具后的结果；
4. AIMessage：模型根据工具结果生成最终回答。

如果 graph 配置了 checkpointer，并且 messages 使用了合适的 reducer，那么这些消息会被保存到对应 thread 的 checkpoint 中。下一次相同 thread_id 调用时，LangGraph 会把历史 messages 加载回来，再把新的 HumanMessage 合并进去。

这就是为什么同一个 thread_id 能记住历史，而不同 thread_id 不会串话。

## 6. 面试回答模板

如果面试官问“LangGraph 的 checkpoint 和 thread_id 是什么关系”，可以这样回答：

> Checkpoint 是图状态的快照，保存某个会话在某一步的 state；thread_id 是会话标识，用来告诉 checkpointer 应该把状态保存到哪条会话链上，或者从哪条会话链恢复。相同 thread_id 会复用历史 checkpoint，因此可以实现多轮记忆；不同 thread_id 对应不同状态链，因此可以实现多会话隔离。本项目中 FastAPI 的 session_id 会映射为 LangGraph 的 thread_id。

## 7. 本项目中的落地方式

本项目的 LangGraph 学习版使用 `InMemorySaver` 实现 checkpoint，并在 `/chat/langgraph` 接口中通过 `session_id` 传递会话标识。当前实现适合求职展示和本地学习，后续如果要生产化，可以把 InMemorySaver 替换为数据库型 checkpointer，并增加会话过期、用户鉴权和历史清理策略。

## 参考来源

- LangGraph Persistence 文档：https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph Memory 文档：https://docs.langchain.com/oss/python/langgraph/add-memory
- LangGraph checkpoint memory 参考：https://reference.langchain.com/python/langgraph.checkpoint/memory
