# Day 04：LangGraph 基础用法 + 状态图式 Agent

> 今日目标：先理解 LangGraph 是用来构建 **有状态 Agent** 的框架，掌握 `State`、`Node`、`Edge`、`Conditional Edge`、`Checkpoint` 五个核心概念；然后把原来的手写 Agent Loop 改造成 LangGraph 状态图。

---

## 目录

1. [一句话总览](#1-一句话总览)
2. [为什么要学 LangGraph](#2-为什么要学-langgraph)
3. [LangGraph 五个核心概念](#3-langgraph-五个核心概念)
4. [手写 Agent Loop vs LangGraph](#4-手写-agent-loop-vs-langgraph)
5. [你项目里的 LangGraph 改造](#5-你项目里的-langgraph-改造)
6. [LangGraph Agent 执行流程](#6-langgraph-agent-执行流程)
7. [Checkpoint 是什么](#7-checkpoint-是什么)
8. [README / 简历怎么写](#8-readme--简历怎么写)
9. [常见面试问题](#9-常见面试问题)
10. [极简背诵版](#10-极简背诵版)

---

# 1. 一句话总览

LangGraph 的作用：

> **LangGraph 用“状态图”的方式编排 Agent，把模型推理、工具调用、条件判断、状态保存变成清晰的 Node 和 Edge。**

你今天的项目改造目标：

```text
原来：手写 while True Agent Loop
现在：LangGraph StateGraph 管理 Agent 流程
```

---

# 2. 为什么要学 LangGraph

手写 Agent Loop 可以跑通 Demo，但复杂后会变得难维护。

手写循环常见问题：

```text
流程逻辑都挤在一个 while 循环里
多节点流程难表达
状态管理容易乱
多轮工具调用难追踪
想加人工审核、分支、记忆会变复杂
```

LangGraph 解决的问题：

```text
用 State 管理 Agent 状态
用 Node 拆分模型、工具、人工审核等步骤
用 Edge 描述流程走向
用 Conditional Edge 做分支判断
用 Checkpoint 保存图执行状态
```

面试表达：

> LangGraph 更适合工程化 Agent，因为它把 Agent 流程从手写循环升级成显式状态图，流程更清晰，也更容易扩展和调试。

---

# 3. LangGraph 五个核心概念

## 3.1 State

State 是图运行时的状态。

可以理解成 Agent 的“工作区”或“上下文记忆”。

常见 State 内容：

```text
messages：对话历史
user_id：用户 ID
retrieved_docs：检索结果
tool_results：工具结果
step_count：执行轮数
```

你项目里使用：

```python
MessagesState
```

它主要保存：

```text
messages
```

---

## 3.2 Node

Node 是图里的一个执行步骤。

每个 Node 通常是一个函数。

你项目里有两个核心 Node：

```text
model 节点：调用模型，决定是否调用工具
tools 节点：执行模型请求的工具
```

代码对应：

```python
def call_model(state):
    ...


def call_tools(state):
    ...
```

---

## 3.3 Edge

Edge 是节点之间的连线，表示流程从哪里走到哪里。

例如：

```python
workflow.add_edge(START, "model")
workflow.add_edge("tools", "model")
```

含义：

```text
图开始后先进入 model 节点
工具执行完后回到 model 节点
```

---

## 3.4 Conditional Edge

Conditional Edge 是条件边。

它根据当前 State 决定下一步去哪。

你项目里的逻辑：

```text
如果模型返回 tool_calls → 去 tools 节点
如果模型没有返回 tool_calls → END 结束
```

代码：

```python
def should_continue(state):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END
```

---

## 3.5 Checkpoint

Checkpoint 是状态保存机制。

它可以保存图每一步的状态。

作用：

```text
支持多轮会话
支持中断恢复
支持调试图执行过程
支持长期记忆的基础能力
```

你项目里用的是内存版：

```python
InMemorySaver()
```

适合 Demo。

生产可以换成：

```text
SQLite
Postgres
Redis
```

---

# 4. 手写 Agent Loop vs LangGraph

## 4.1 手写 Agent Loop

你原来的流程：

```python
while True:
    response = llm.invoke(messages, tools=TOOLS)

    if not response.tool_calls:
        print(response.content)
        break

    execute_tools(response.tool_calls)
```

优点：

```text
简单直接
适合学习原理
```

缺点：

```text
流程复杂后难维护
状态管理不清晰
分支逻辑容易混乱
```

---

## 4.2 LangGraph

LangGraph 把流程拆成图：

```text
START
  ↓
model node
  ↓
条件判断：是否有 tool_calls
  ├── 有 → tools node → model node
  └── 无 → END
```

优点：

```text
节点清晰
流程可视化思维强
状态统一管理
适合扩展复杂 Agent
支持 checkpoint
```

---

# 5. 你项目里的 LangGraph 改造

新增文件：

```text
langgraph_agent.py
LangGraph_RAG_Agent_demo.py
Day04_LangGraph_Basic_Agent_Notes.md
```

## 5.1 `langgraph_agent.py`

负责 LangGraph Agent 核心逻辑：

```text
StateGraph
model node
tools node
conditional edge
checkpoint
run_langgraph_agent()
```

## 5.2 `LangGraph_RAG_Agent_demo.py`

新的运行入口：

```bash
python LangGraph_RAG_Agent_demo.py
```

它和旧入口类似，但内部调用的是：

```python
run_langgraph_agent(query)
```

## 5.3 复用已有工具

LangGraph 版本没有重写工具，而是复用了已有：

```text
search_docs
calculator
get_weather
```

对应文件：

```text
tools.py
schemas.py
models.py
vector_store.py
```

---

# 6. LangGraph Agent 执行流程

当前图结构：

```text
START
  ↓
model
  ↓
should_continue 判断
  ├── 如果有 tool_calls → tools
  │                         ↓
  │                       model
  │
  └── 如果没有 tool_calls → END
```

完整运行过程：

```text
用户输入问题
  ↓
进入 model 节点
  ↓
模型决定是否调用工具
  ↓
如果调用工具，进入 tools 节点
  ↓
tools 节点执行 search_docs / calculator / get_weather
  ↓
工具结果以 ToolMessage 放回 State
  ↓
回到 model 节点
  ↓
模型基于工具结果继续判断或最终回答
  ↓
没有 tool_calls 后结束
```

---

# 7. Checkpoint 是什么

你当前代码：

```python
checkpointer = InMemorySaver()
graph_app = workflow.compile(checkpointer=checkpointer)
```

运行时传入：

```python
config = {
    "configurable": {"thread_id": thread_id},
    "recursion_limit": 10,
}
```

其中：

```text
thread_id：区分不同会话
recursion_limit：防止图无限循环
```

面试表达：

> Checkpoint 可以保存 LangGraph 每一步执行状态，让 Agent 支持多轮会话、中断恢复和状态追踪。Demo 中我用 InMemorySaver，生产中可以换成持久化存储。

---

# 8. README / 简历怎么写

## README 写法

可以写：

> 项目使用 LangGraph 将原本手写的 Agent Loop 改造成状态图结构。通过 `StateGraph` 管理对话状态，定义 `model` 节点负责模型推理和工具调用决策，定义 `tools` 节点执行 `search_docs`、`calculator`、`get_weather` 等工具，并通过 Conditional Edge 判断是否继续调用工具或结束回答。同时使用 Checkpoint 保存图状态，使 Agent 流程更加清晰、可维护、可扩展。

## 简历写法

可以写：

> 使用 LangGraph 重构 RAG Agent，将模型推理、工具调用、条件分支和最终回答流程抽象为 StateGraph，复用 search_docs、calculator、weather 等工具，实现可维护的状态图式 Agent 编排。

---

# 9. 常见面试问题

## Q1：LangGraph 是做什么的？

回答：

> LangGraph 是用于构建有状态 Agent 的框架。它用 StateGraph 表达 Agent 流程，把模型调用、工具调用、条件分支、状态保存拆成节点和边，更适合复杂 Agent 的工程化编排。

---

## Q2：State 是什么？

回答：

> State 是图运行过程中的状态，类似 Agent 的工作区。常见 State 包括 messages、工具结果、检索结果、用户信息等。每个节点读取 State，并返回对 State 的更新。

---

## Q3：Node 是什么？

回答：

> Node 是图中的一个执行步骤，通常是一个函数。例如 model node 负责调用模型，tools node 负责执行工具。

---

## Q4：Conditional Edge 是什么？

回答：

> Conditional Edge 是条件边，它根据当前 State 决定下一步走向。例如模型返回 tool_calls 时进入工具节点，否则结束。

---

## Q5：Checkpoint 有什么用？

回答：

> Checkpoint 用来保存图执行状态，支持多轮会话、中断恢复、调试和长期状态管理。Demo 可以用 InMemorySaver，生产可以换成持久化存储。

---

## Q6：LangGraph 和手写 Agent Loop 有什么区别？

回答：

> 手写 Agent Loop 是用 while 循环控制流程，适合简单 Demo；LangGraph 用状态图表达流程，节点、边、条件分支更清晰，也更容易扩展复杂 Agent，比如人工审核、多工具、多分支和状态恢复。

---

# 10. 极简背诵版

1. **LangGraph 是用状态图构建有状态 Agent 的框架。**
2. **State 是 Agent 的运行状态，最常见的是 messages。**
3. **Node 是图里的执行步骤，比如 model node、tools node。**
4. **Edge 表示节点之间的固定流转。**
5. **Conditional Edge 根据状态决定下一步去哪。**
6. **Checkpoint 用来保存图执行状态，支持会话记忆和中断恢复。**
7. **LangGraph 可以替代手写 Agent Loop，让流程更清晰、可维护、可扩展。**
8. **当前项目中，model 节点负责模型决策，tools 节点负责执行 search_docs / calculator / get_weather。**
9. **如果模型返回 tool_calls，就进入 tools 节点；否则进入 END。**
10. **生产中可以把 InMemorySaver 换成持久化 checkpointer。**
