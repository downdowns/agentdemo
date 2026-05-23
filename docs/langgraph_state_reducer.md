# LangGraph State、Reducer 与 add_messages

> 本文是学习型知识库文档，基于 LangGraph 官方 Graph API、message reducer 参考文档和本项目实践整理，用于 RAG 检索与 chunk-level 评估实验。

## 1. State 是什么

LangGraph 中的 State 是图运行时共享的数据结构。每个节点执行时，会读取当前 state，并返回一部分新的 state 更新。LangGraph 根据字段的 reducer 规则，把节点返回的新值合并到旧 state 中。

在 Agent 项目中，State 可以理解为“Agent 工作流的上下文容器”。它通常包含：

- `messages`：对话历史，包括用户消息、模型消息、工具结果消息；
- `tool_calls`：工具调用记录，用于调试、日志和结构化返回；
- `sources`：RAG 检索命中的来源文档；
- `rounds`：循环轮数或调试字段；
- 其他业务字段，例如用户 ID、权限、检索参数等。

手写 Agent Loop 通常用一个 Python 列表保存 messages，并在 if/else 循环里追加工具结果。LangGraph 则把这些变量抽象成 State，让节点之间通过状态更新协作。

## 2. 为什么不能只用 MessagesState

`MessagesState` 很适合最简单的聊天机器人，因为它只维护 messages 字段。但真实 Agent 往往不仅需要 messages，还需要额外结构化信息。

例如本项目的 API 返回值需要包含：

```json
{
  "answer": "...",
  "tool_calls": [...],
  "sources": [...],
  "messages_count": 8
}
```

如果只用 MessagesState，`tool_calls` 和 `sources` 就没有独立字段，不方便跨节点累积，也不方便在 API 层结构化返回。因此本项目定义了自定义 `GraphState`，在 messages 之外增加 tool_calls 和 sources。

## 3. Annotated[list, add_messages] 是什么意思

在 LangGraph 中，State 字段可以用 `Annotated` 指定 reducer。例如：

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
```

这表示：

- `messages` 的类型是 list；
- 当节点返回新的 messages 时，不是直接覆盖旧 messages；
- 而是使用 `add_messages` 这个 reducer 把新消息合并到旧消息中。

如果没有 reducer，LangGraph 默认会对字段进行覆盖式更新。对于 messages 来说，覆盖会导致历史消息丢失，因此必须使用 `add_messages` 或类似的合并逻辑。

## 4. Reducer 是什么

Reducer 是“旧状态 + 新更新 -> 合并后的状态”的函数。它决定同一个字段在多次节点更新时如何合并。

可以用一个简单公式理解：

```text
new_state[field] = reducer(old_state[field], node_output[field])
```

不同字段需要不同 reducer：

- messages：通常追加新消息，但也要处理消息 ID 覆盖等情况；
- tool_calls：通常把新的工具调用记录追加到旧列表；
- sources：通常追加并去重，避免同一个文档来源重复出现；
- counter：可能需要累加；
- latest_answer：可能应该覆盖。

Reducer 的设计决定了 state 是否能正确跨节点、跨轮次累积。

## 5. add_messages 做了什么

`add_messages` 是 LangGraph 用于消息列表合并的 reducer。它的核心作用是把节点返回的新 message 合并到已有 messages 中，而不是简单覆盖整个列表。

在 Agent Loop 中，一个典型过程是：

1. 用户输入加入 messages；
2. 模型节点返回 AIMessage；
3. 工具节点返回 ToolMessage；
4. 模型节点再次返回最终 AIMessage。

如果 messages 没有 reducer，每个节点返回的新 messages 可能会覆盖之前的消息，导致上下文断裂。使用 add_messages 后，LangGraph 会保留历史消息，并追加新的消息，使模型在下一轮调用时仍能看到完整上下文。

## 6. 为什么 tool_calls 和 sources 也需要 reducer

本项目中 `tool_calls` 和 `sources` 也是 state 字段。如果节点多次返回这些字段，但没有 reducer，LangGraph 默认会用新值覆盖旧值。

例如第一轮调用了 calculator：

```python
tool_calls = [{"name": "calculator"}]
```

第二轮调用了 search_docs：

```python
tool_calls = [{"name": "search_docs"}]
```

如果没有 reducer，第二轮结果会覆盖第一轮，最终只剩 search_docs。这样 API 返回的工具调用记录就不完整。

因此本项目实现了类似：

```python
def merge_tool_calls(old, new):
    return old + new
```

用于跨轮累积工具调用记录。

## 7. merge_sources 为什么要去重

RAG 检索经常会多次命中同一个文档来源。例如用户连续追问 LangGraph checkpoint，系统可能每次都检索到 `langgraph_persistence.md`。如果 sources 只是简单追加，最终可能变成：

```python
["langgraph_persistence.md", "langgraph_persistence.md", "langgraph_persistence.md"]
```

这对前端展示和日志分析都不友好。因此 sources reducer 通常需要去重，保留首次出现顺序：

```python
["langgraph_persistence.md"]
```

去重后的 sources 更适合作为“本轮或当前会话引用过哪些知识来源”的摘要。

## 8. 本项目中的状态设计

本项目 LangGraph 版本的状态设计可以概括为：

```python
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls: Annotated[list, merge_tool_calls]
    sources: Annotated[list, merge_sources]
```

含义是：

- messages 用 add_messages 维护对话上下文；
- tool_calls 用 merge_tool_calls 记录工具调用轨迹；
- sources 用 merge_sources 记录 RAG 来源并去重。

这让 LangGraph Agent 的返回结构更接近手写 `agent.py`，同时保留了图工作流的可扩展性。

## 9. 面试回答模板

如果面试官问“reducer 在 LangGraph 中有什么作用”，可以回答：

> Reducer 决定节点返回的新 state 如何和旧 state 合并。比如 messages 需要用 add_messages 追加历史消息，否则新消息会覆盖旧消息；tool_calls 需要自定义 reducer 累积工具调用记录；sources 需要自定义 reducer 追加并去重。没有 reducer 的字段通常会被覆盖，因此在有状态多轮 Agent 中，reducer 是保证上下文和结构化记录正确累积的关键机制。

## 参考来源

- LangGraph Graph API 文档：https://docs.langchain.com/oss/python/langgraph/use-graph-api
- LangGraph message 参考：https://reference.langchain.com/python/langgraph/graph/message
