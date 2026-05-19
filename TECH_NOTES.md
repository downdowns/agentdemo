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

