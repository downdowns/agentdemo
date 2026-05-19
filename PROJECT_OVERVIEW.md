# Enterprise RAG Agent 项目说明文档

> 当前项目阶段：RAG + Function Calling Agent + FastAPI 服务化 + 最小评估脚本。  
> 当前定位：从学习 demo 升级为可展示、可讲解、可继续扩展的 Agent 项目雏形。

---

## 1. 项目简介

本项目实现了一个基于 **RAG（Retrieval-Augmented Generation，检索增强生成）** 和 **Function Calling / Tool Calling** 的多工具 Agent。

系统可以根据用户问题自动判断是否需要调用工具：

- 本地知识库检索：`search_docs`
- 数学计算：`calculator`
- 天气查询：`get_weather`

Agent 调用工具后，会把工具结果返回给大模型，由大模型基于工具结果生成最终回答。

目前系统已经支持：

1. 本地文档知识库加载
2. 文档切分
3. Embedding 向量化
4. Chroma 向量数据库持久化
5. RAG 检索工具
6. Function Calling Agent Loop
7. 工具调用记录
8. RAG 来源追踪
9. Agent 执行日志
10. 最小 Agent 评估脚本
11. FastAPI HTTP 接口
12. Swagger 接口文档

---

## 2. 当前项目架构

```text
用户 / curl / Swagger / 未来前端
        ↓
FastAPI /chat
        ↓
run_agent()
        ↓
LLM 判断是否需要工具
        ↓
工具层
  ├── search_docs：本地知识库检索
  ├── calculator：数学计算
  └── get_weather：模拟天气查询
        ↓
工具结果写回 messages
        ↓
LLM 基于工具结果生成最终回答
        ↓
结构化 JSON 返回
        ↓
写入 logs/agent.log
```

---

## 3. 核心文件说明

### 3.1 配置文件

#### `config.py`

负责项目全局配置：

- `.env` 加载
- 本地知识库目录
- Chroma 数据库目录
- Chroma collection 名称
- 文档状态签名文件
- Agent 最大循环轮数

关键配置：

```python
DOCS_DIR = "./docs"
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "rag_demo_collection"
MAX_AGENT_ROUNDS = 5
```

---

### 3.2 模型初始化

#### `models.py`

负责初始化：

1. 聊天模型 `llm`
2. Embedding 模型 `embeddings`

当前聊天模型使用 `ChatOpenAI` 调用 DeepSeek 的 OpenAI-compatible API。

当前 Embedding 模型：

```text
BAAI/bge-m3
```

用途：

- 聊天模型：负责判断工具调用和生成最终回答
- Embedding 模型：负责把文档和 query 转成向量，用于语义检索

---

### 3.3 向量库与 RAG

#### `vector_store.py`

负责完整的本地知识库处理流程：

```text
读取 docs 文件夹
  ↓
加载 .md / .txt 文档
  ↓
包装成 LangChain Document
  ↓
RecursiveCharacterTextSplitter 切分
  ↓
使用 bge-m3 生成向量
  ↓
写入 / 加载 Chroma
```

当前文档切分参数：

```python
chunk_size = 100
chunk_overlap = 20
```

当前支持文件类型：

- `.md`
- `.txt`

项目还实现了一个文档状态签名机制：

- 如果 `docs/` 里的文档没有变化，直接加载已有 Chroma 数据库
- 如果文档发生变化，删除旧数据库并重新构建

这样可以避免每次启动都重新向量化。

---

### 3.4 工具函数

#### `tools.py`

实现 Agent 可以调用的真实 Python 工具：

#### `search_docs(query: str, k: int = 2)`

搜索本地知识库。

返回结构：

```python
[
    {
        "source": "rag_notes.md",
        "content": "检索到的文档片段"
    }
]
```

#### `calculator(operation: str, a: float, b: float)`

执行基础数学运算。

支持：

- add
- subtract
- multiply
- divide

#### `get_weather(city: str)`

查询模拟天气。

注意：当前是 demo 模拟数据，不是真实天气 API。

---

### 3.5 工具 Schema

#### `schemas.py`

定义给模型看的工具说明书。

模型不会直接看到 Python 函数，而是看到类似这样的 schema：

```python
{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "执行基础数学运算",
        "parameters": {...}
    }
}
```

工具 schema 的作用：

1. 告诉模型有哪些工具
2. 告诉模型工具能做什么
3. 告诉模型需要传哪些参数
4. 约束模型生成结构化工具调用

---

### 3.6 Agent 主循环

#### `agent.py`

这是当前项目最核心的文件。

它实现了手写 Agent Loop：

```text
用户输入
  ↓
构造 messages
  ↓
调用模型
  ↓
判断是否有 tool_calls
  ├── 没有：返回最终答案
  └── 有：执行工具
        ↓
      工具结果追加到 messages
        ↓
      再次调用模型
```

当前 `run_agent()` 返回结构：

```python
{
    "user_query": "用户问题",
    "answer": "最终回答",
    "tool_calls": [
        {
            "name": "工具名",
            "args": {},
            "result": {}
        }
    ],
    "sources": ["知识库来源文件"],
    "rounds": 2
}
```

已经实现的工程化能力：

- 结构化返回
- 工具调用记录
- RAG 来源提取
- 工具异常保护
- 最大轮数限制
- JSONL 日志写入

---

### 3.7 命令行入口

#### `RAG_Agent_demo.py`

提供命令行交互入口。

用户可以在终端输入问题，系统调用 `run_agent()` 并打印结构化结果。

---

### 3.8 FastAPI 服务

#### `app/main.py`

把 Agent 封装成 HTTP API。

当前接口：

#### `GET /health`

健康检查接口。

返回：

```json
{
  "status": "ok",
  "service": "rag-agent-api"
}
```

#### `POST /chat`

Agent 聊天接口。

请求：

```json
{
  "message": "帮我计算 23 乘以 19"
}
```

返回：

```json
{
  "user_query": "帮我计算 23 乘以 19",
  "answer": "23 乘以 19 的结果是 437。",
  "tool_calls": [...],
  "sources": [],
  "rounds": 2
}
```

启动方式：

```bash
uvicorn app.main:app --reload
```

接口文档：

```text
http://127.0.0.1:8000/docs
```

---

### 3.9 Agent 评估

#### `eval/questions.json`

保存评估问题和期望工具。

示例：

```json
{
  "id": "calc_001",
  "question": "帮我计算 23 乘以 19",
  "expected_tools": ["calculator"]
}
```

#### `eval/run_eval.py`

自动执行评估集：

1. 读取 `questions.json`
2. 调用 `run_agent()`
3. 从 `tool_calls` 中提取实际调用工具
4. 和 `expected_tools` 对比
5. 输出通过率

当前评估重点：

```text
工具调用是否正确
```

---

## 4. 当前运行方式

### 4.1 命令行运行

```bash
python RAG_Agent_demo.py
```

### 4.2 FastAPI 运行

```bash
uvicorn app.main:app --reload
```

测试：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "RAG 是什么？"}'
```

### 4.3 运行评估脚本

```bash
python eval/run_eval.py
```

---

## 5. 当前项目亮点

1. 不只是普通 RAG，而是把 RAG 封装成 Agent 工具
2. Agent 可以根据问题自动选择工具
3. 支持多工具调用
4. 支持工具异常处理
5. 支持结构化返回
6. 支持 RAG 来源追踪
7. 支持本地日志记录
8. 支持最小工具调用评估
9. 支持 FastAPI 服务化
10. 支持 Swagger 文档调试

---

## 6. 当前不足与下一步计划

### 6.1 当前不足

1. `/chat` 还没有做空输入校验
2. `/chat` 还没有统一异常处理
3. RAG 还没有 reranker
4. 评估只评估了工具调用，没有评估答案质量
5. 天气工具仍然是模拟数据
6. 还没有 Docker 部署
7. 还没有 LangGraph 工作流版本的工程化封装
8. 还没有前端页面

### 6.2 下一步计划

优先级从高到低：

1. FastAPI 请求校验和错误处理
2. README 项目包装
3. 增加答案质量评估
4. 增加 RAG source 命中评估
5. 增加 rerank
6. Docker 化
7. LangGraph 重构 Agent Loop
8. vLLM 部署本地 Qwen 小模型

