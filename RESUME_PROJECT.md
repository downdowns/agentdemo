# 简历项目描述：企业知识库 RAG Agent 系统

> 项目名称可选：
> - 企业知识库 RAG Agent 系统
> - 基于 LangGraph 的多工具 RAG Agent 服务
> - Enterprise RAG Agent

---

## 版本一：简历精简版

**企业知识库 RAG Agent 系统**  
技术栈：Python、FastAPI、LangChain、LangGraph、Chroma、BGE-M3、Function Calling

- 基于 LangChain + Chroma + BGE-M3 构建本地知识库 RAG 流程，支持 Markdown/TXT 文档加载、文本切分、向量化入库、语义检索和答案来源追踪。
- 手写 Function Calling Agent Loop，将知识库检索、计算器和天气查询封装为工具，支持模型自动选择工具、多工具调用、工具异常处理和最大轮数控制。
- 设计结构化返回格式，输出 `answer`、`tool_calls`、`sources`、`rounds` 等字段，并使用 JSONL 记录 Agent 执行日志，便于调试和追踪。
- 使用 LangGraph 重构 Agent 流程，将模型节点、工具节点和条件边显式建模为状态图；通过 checkpoint 和 thread_id/session_id 实现有状态多轮对话与多会话隔离。
- 基于 FastAPI 提供 `/chat` 和 `/chat/langgraph` 接口，并编写最小评估脚本，自动验证计算、天气、RAG 和混合问题下的工具调用正确性。

---

## 版本二：面试官更容易看懂版

**企业知识库 RAG Agent 系统**  
项目描述：面向企业文档问答场景，实现一个支持知识库检索、工具调用、多轮会话和 Agent 评估的 LLM 应用服务。

主要工作：

1. 使用 `BAAI/bge-m3` embedding 和 Chroma 构建本地向量知识库，完成文档加载、切分、向量化、持久化和相似度检索。
2. 将 RAG 检索封装为 `search_docs` 工具，并额外实现 `calculator`、`get_weather` 工具，通过 Function Calling 让模型自动判断是否调用工具。
3. 手写 Agent Loop，完成 `messages → tool_calls → 工具执行 → ToolMessage → 最终回答` 的闭环，并加入工具异常处理、最大轮数限制和结构化返回。
4. 引入 LangGraph，将手写 Agent Loop 重构为 `model node → tools node → conditional edge` 的状态图，并使用 checkpoint + thread_id 实现会话记忆和多用户隔离。
5. 使用 FastAPI 封装服务接口，提供 `/chat` 手写 Agent 接口和 `/chat/langgraph` 有状态 LangGraph Agent 接口，支持 Swagger 调试和 HTTP 错误处理。
6. 构建 `eval/questions.json` 和 `eval/run_eval.py` 最小评估集，自动对比 expected tools 与 actual tools，验证 Agent 工具调用准确性。

---

## 版本三：适合写在简历项目经历里的 4 条

- 构建企业知识库 RAG 流程：基于 BGE-M3 embedding 和 Chroma 向量库，实现文档加载、切分、向量化、相似度检索和答案来源追踪。
- 实现多工具 Function Calling Agent：封装 `search_docs`、`calculator`、`get_weather` 等工具，手写 Agent Loop 支持自动工具选择、多工具调用、异常处理和最大轮数控制。
- 使用 LangGraph 重构 Agent 工作流：通过 StateGraph 显式建模 model/tools 节点和 conditional edge，并结合 checkpoint、thread_id/session_id 实现有状态对话和多会话隔离。
- 完成 FastAPI 服务化与评估：提供 `/chat` 和 `/chat/langgraph` 接口，支持结构化 JSON 返回、请求校验、错误处理，并编写最小评估脚本验证工具调用正确率。

---

## 面试口述版：1 分钟介绍

我做了一个企业知识库 RAG Agent 系统。底层用本地 Markdown/TXT 文档构建 Chroma 向量库，使用 BGE-M3 做 embedding，实现文档切分、向量检索和来源追踪。Agent 层我先手写了 Function Calling Agent Loop，让模型根据用户问题自动调用 `search_docs`、`calculator`、`get_weather` 等工具，并把工具结果返回给模型生成最终回答。之后我用 LangGraph 把这个流程重构成状态图，加入 checkpoint 和 thread_id/session_id，实现有状态多轮对话和多会话隔离。工程化方面，我用 FastAPI 提供 `/chat` 和 `/chat/langgraph` 接口，做了结构化返回、异常处理、日志记录和最小评估脚本。

---

## 面试可展开的技术点

### RAG

- 文档加载：读取 `docs/` 下 `.md` / `.txt`
- 文档切分：`RecursiveCharacterTextSplitter`
- Embedding：`BAAI/bge-m3`
- 向量库：Chroma
- 检索工具：`search_docs`
- 来源追踪：`sources`

### Agent

- Tool schema
- Function Calling
- `message.tool_calls`
- 工具映射 `AVAILABLE_FUNCTIONS`
- `ToolMessage`
- 最大轮数限制
- 工具异常处理

### LangGraph

- `GraphState`
- `add_messages`
- 自定义 reducer：`merge_tool_calls` / `merge_sources`
- `model node`
- `tools node`
- `conditional edge`
- checkpoint
- thread_id / session_id

### 工程化

- FastAPI `/chat`
- FastAPI `/chat/langgraph`
- Pydantic 请求体
- HTTP 400 / 500
- JSONL 日志
- eval 脚本
- GitHub README 和项目文档

---

## 简历关键词

```text
RAG、Function Calling、Tool Calling、Agent Loop、LangGraph、FastAPI、
Chroma、BGE-M3、Embedding、Vector Database、Checkpoint、thread_id、
多会话隔离、Agent Evaluation、JSONL Logging
```
