# RAG + Function Calling 面试补充笔记

> 目的：你已经通过动手做 Demo 入门了，但面试时可能会被问到一些概念、工程细节和生产问题。这个文件帮你补齐需要理解和背熟的内容。

---

## 1. 你现在已经掌握了什么

你目前的项目已经包含：

- DeepSeek 聊天模型调用
- HuggingFace Embedding：`BAAI/bge-m3`
- Chroma 向量数据库
- 本地 `.md` / `.txt` 文档加载
- 文本切分
- 向量库持久化
- 文档变化自动重建向量库
- RAG 检索问答
- Function Calling
- Agent Loop
- 多工具调用：
  - `search_docs`
  - `calculator`
  - `get_weather`
- 代码模块化拆分：
  - `config.py`
  - `models.py`
  - `vector_store.py`
  - `tools.py`
  - `schemas.py`
  - `agent.py`
  - `RAG_Agent_demo.py`

这已经算是 RAG + Agent 的入门项目。

但面试时还需要补充一些理论和工程知识。

---

# Part 1：Function Calling 需要补的概念

## 1.1 Function Calling 是什么

Function Calling 是让大模型输出结构化的工具调用请求，而不是直接执行代码。

模型本身不会真的调用函数，它只会生成类似这样的结构：

```json
{
  "name": "get_weather",
  "arguments": {
    "city": "上海"
  }
}
```

真正执行函数的是你的程序。

流程：

```text
用户问题
  ↓
模型判断是否需要调用工具
  ↓
模型输出工具名和参数
  ↓
Python 程序解析工具名和参数
  ↓
Python 执行真实函数
  ↓
把工具结果返回给模型
  ↓
模型生成最终回答
```

---

## 1.2 Function Calling 里的三个核心对象

### 1. 真实函数

这是 Python 真正执行的函数：

```python
def get_weather(city: str):
    return {"city": city, "weather": "晴天"}
```

### 2. Tool Schema

这是给模型看的工具说明书：

```python
get_weather_schema = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称"
                }
            },
            "required": ["city"]
        }
    }
}
```

### 3. 函数映射表

这是给程序看的：

```python
AVAILABLE_FUNCTIONS = {
    "get_weather": get_weather,
}
```

模型只会说它要调用 `get_weather`，Python 需要通过这个字典找到真正的函数。

---

## 1.3 Tool Schema 为什么重要

Tool Schema 决定模型能不能正确调用工具。

Schema 里最重要的是：

- `name`：工具名称
- `description`：什么时候应该用这个工具
- `parameters`：工具需要哪些参数
- `required`：哪些参数必须传

如果 description 写得不清楚，模型可能：

- 不调用工具
- 调错工具
- 参数传错
- 把自然语言当成参数乱传

---

## 1.4 `tool_choice="auto"` 是什么意思

```python
tool_choice="auto"
```

意思是让模型自己决定：

```text
要不要调用工具
调用哪个工具
调用几次工具
```

如果用户问：

```text
15 加 27 等于多少？
```

模型可能调用：

```text
calculator
```

如果用户问：

```text
你好
```

模型可能不调用工具，直接回答。

---

## 1.5 Function Calling 和普通 Prompt 的区别

普通 Prompt：

```text
用户问：上海天气怎么样？
模型直接回答：上海今天晴天。
```

问题：模型可能胡编。

Function Calling：

```text
用户问：上海天气怎么样？
模型调用 get_weather
程序返回真实天气或模拟天气
模型基于工具结果回答
```

优势：

- 可以接真实系统
- 可以查数据库
- 可以调用 API
- 可以计算
- 可以减少模型胡编

---

## 1.6 Agent Loop 是什么

普通 Function Calling 通常只处理一轮：

```text
模型调用工具
程序执行工具
模型回答
结束
```

Agent Loop 可以处理多轮：

```text
模型调用工具
程序执行工具
模型继续判断是否还要调用工具
如果还要，继续执行
直到模型不再调用工具
输出最终回答
```

适合复杂任务。

例如用户问：

```text
查一下 RAG 的核心流程，再计算 15 加 27，然后告诉我上海天气。
```

Agent 可能调用：

```text
第 1 轮：search_docs
第 2 轮：calculator
第 3 轮：get_weather
第 4 轮：最终回答
```

或者一轮里同时调用多个工具。

---

## 1.7 为什么模型有时候一轮只调用一个工具

这是模型的策略，不是代码错误。

原因可能是：

- 模型更保守
- 模型想先拿到一个工具结果再决定下一步
- 工具之间可能有依赖关系
- 模型没有选择并行调用

你的代码里有：

```python
for tool_call in message.tool_calls:
```

这说明代码支持一轮多个工具调用。

---

## 1.8 Function Calling 面试常见问题

### Q1：模型会真的执行函数吗？

不会。模型只生成工具调用请求，真正执行函数的是你的程序。

### Q2：为什么需要 tool_call_id？

当模型一次调用多个工具时，`tool_call_id` 用来对应：

```text
哪个工具调用 对应 哪个工具结果
```

返回工具结果时必须带上它。

### Q3：工具参数不合法怎么办？

应该做防御性处理：

- 检查工具名是否存在
- 检查参数是否完整
- 捕获异常
- 返回结构化错误信息给模型

### Q4：Function Calling 能解决幻觉吗？

只能减少一部分幻觉，不能完全解决。

它可以让模型基于真实工具结果回答，但如果工具结果本身不准，或者模型没有正确使用工具，仍然可能出错。

---

# Part 2：RAG 需要补的概念

## 2.1 RAG 是什么

RAG = Retrieval-Augmented Generation，检索增强生成。

它的核心思想：

```text
不要只依赖模型参数里的知识
而是先从外部知识库检索相关资料
再让模型基于资料回答
```

流程：

```text
文档加载
  ↓
文本切分
  ↓
Embedding 向量化
  ↓
存入向量数据库
  ↓
用户提问
  ↓
问题向量化
  ↓
向量数据库检索相关片段
  ↓
把片段作为上下文交给大模型
  ↓
模型回答
```

---

## 2.2 RAG 解决什么问题

RAG 主要解决：

- 模型不知道私有知识
- 模型知识过期
- 需要引用公司文档
- 需要减少幻觉
- 不想为了少量知识重新训练模型

---

## 2.3 RAG 和微调的区别

### RAG

适合：

- 知识经常变化
- 需要查文档
- 需要引用来源
- 企业知识库问答

优点：

- 更新知识方便
- 不需要训练模型
- 可追溯来源

缺点：

- 依赖检索质量
- 文档处理复杂
- 上下文长度有限

### 微调

适合：

- 学习固定风格
- 学习固定格式
- 学习稳定任务模式

优点：

- 推理时不一定需要检索
- 输出风格更稳定

缺点：

- 更新知识成本高
- 不适合频繁变化的知识
- 不适合大量事实型私有知识

面试回答：

> 如果是企业知识库问答，我优先用 RAG；如果是固定输出格式或领域风格适配，可以考虑微调；复杂场景也可以 RAG + 微调结合。

---

## 2.4 Embedding 是什么

Embedding 是把文本转成向量。

例如：

```text
RAG 的核心流程是什么？
```

会变成：

```python
[0.01, -0.23, 0.87, ...]
```

向量之间可以计算相似度。

语义相近的文本，向量距离更近。

---

## 2.5 向量数据库是什么

向量数据库用来存储和检索向量。

它通常存：

- 文本片段
- 向量
- metadata
- id

常见向量数据库：

- Chroma
- FAISS
- Milvus
- Pinecone
- Weaviate
- Qdrant
- Elasticsearch / OpenSearch 向量检索

你的项目用的是 Chroma。

---

## 2.6 Chunk 是什么

Chunk 是文档切分后的小片段。

为什么要切分？

- 文档太长，不能整篇放进上下文
- 检索粒度太粗会不准
- 模型上下文长度有限

常见参数：

```python
chunk_size=100
chunk_overlap=20
```

生产中 chunk_size 需要调优。

---

## 2.7 chunk_size 怎么选

没有固定答案，取决于文档类型。

一般原则：

- FAQ / 短文本：小 chunk
- 技术文档：中等 chunk
- 法律/合同：保留完整条款
- 表格：尽量保留结构
- 代码：按函数或类切分

如果 chunk 太小：

- 上下文不完整
- 模型拿不到足够信息

如果 chunk 太大：

- 检索不精准
- token 成本高
- 噪音多

---

## 2.8 chunk_overlap 为什么需要

重叠是为了防止一句话或一个概念被切断。

例如：

```text
RAG 的核心流程包括加载文档、文本切分、Embedding 向量化...
```

如果正好从中间切开，语义会断。

`chunk_overlap` 可以保留上下文连续性。

---

## 2.9 top_k 是什么

`top_k` 表示检索最相关的前 k 个片段。

例如：

```python
similarity_search(query, k=3)
```

表示返回最相关的 3 个文档片段。

k 太小：可能漏掉答案。

k 太大：上下文噪音多，成本高，模型容易被干扰。

---

## 2.10 RAG 为什么会回答错

RAG 出错通常分两类。

### 1. 检索错了

模型拿到的上下文本身不相关。

解决：

- 调整 chunk_size
- 调整 top_k
- 换 embedding 模型
- Query Rewrite
- Hybrid Search
- Rerank
- 加 metadata filter

### 2. 检索对了，但生成错了

模型没有正确基于上下文回答。

解决：

- 优化 prompt
- 要求只基于上下文回答
- 加引用来源
- 限制回答格式
- 降低 temperature
- 增加拒答逻辑

面试回答重点：

> 先看 retrieved chunks。如果检索结果不对，先优化检索；如果检索结果对但回答错，再优化生成。

---

# Part 3：生产级 RAG 必须知道的问题

## 3.1 检索质量优化

基础向量检索只是第一步。

生产中常用优化：

### 1. Query Rewrite

把用户问题改写成更适合检索的问题。

例如用户问：

```text
这个怎么弄？
```

结合上下文改写成：

```text
RAG 向量数据库如何持久化？
```

### 2. Multi-query Retrieval

让模型生成多个不同问法，然后分别检索。

适合用户问题表达不清楚时。

### 3. Hybrid Search

关键词检索 + 向量检索。

向量检索擅长语义相似。

关键词检索擅长精确匹配：

- 产品型号
- 人名
- 编号
- 错误码
- API 名称

### 4. Rerank

先召回 top 20，再用 reranker 重排，选 top 3 给模型。

典型流程：

```text
向量检索 top 20
  ↓
reranker 重排序
  ↓
取 top 3
  ↓
给 LLM 回答
```

Rerank 是生产 RAG 很重要的优化点。

---

## 3.2 相似度阈值

如果检索结果分数太低，就应该拒答。

否则模型可能根据不相关上下文胡编。

逻辑：

```text
如果最高相似度低于阈值：
    回答：根据当前知识库无法回答
```

你的项目下一步可以加这个。

---

## 3.3 引用来源

生产 RAG 最好返回来源：

```text
答案：RAG 的核心流程包括...

参考来源：
- rag_notes.md
```

好处：

- 用户可以验证
- 降低幻觉风险
- 方便 debug
- 更像真实产品

---

## 3.4 增量更新

你现在的方案是：

```text
docs 变化 → 删除整个 chroma_db → 全量重建
```

Demo 可以这样。

生产不应该这样。

生产应该：

```text
每个文档有 doc_id
每个 chunk 有 chunk_id
某个文档变了，只删除这个文档对应的 chunks
然后重新插入这个文档的新 chunks
```

面试回答：

> 大规模知识库不会全量重建。我会维护文档元数据和 chunk_id，基于文件 hash 或更新时间做增量更新。

---

## 3.5 权限控制

企业 RAG 一定会涉及权限。

例如：

- 销售只能看销售资料
- HR 只能看 HR 资料
- 管理层文档普通员工不能看

做法：

```text
metadata 里存权限信息
检索时加 filter
只召回用户有权限的文档
```

重要原则：

> 权限过滤应该发生在检索阶段，而不是检索后靠 prompt 要求模型不要说。

错误做法：

```text
先检索所有文档，再让模型不要回答没权限内容
```

因为模型已经看到了敏感信息。

---

## 3.6 成本和延迟

生产中要考虑：

- embedding 成本
- LLM token 成本
- 检索耗时
- rerank 耗时
- 并发压力

优化方法：

- embedding 缓存
- 文档增量更新
- 控制 top_k
- rerank 只对候选集做
- 热门问题缓存
- 流式输出
- 小模型处理简单任务
- 异步批量 embedding

---

## 3.7 日志和可观测性

生产 RAG 必须记录日志。

建议记录：

- 用户问题
- 改写后的 query
- 检索到的 chunks
- similarity score
- 最终 prompt
- 模型回答
- 用户反馈
- latency
- token usage
- error trace

面试回答：

> RAG 优化不能只凭感觉，需要记录 query、retrieved chunks、answer 和用户反馈，定位是检索问题还是生成问题。

---

# Part 4：RAG 评估

## 4.1 为什么需要评估

不能只靠感觉判断 RAG 好不好。

需要构造测试集。

测试集通常包含：

```text
问题
标准答案
标准引用文档
可接受答案
```

---

## 4.2 检索评估指标

### Recall@K

标准答案所在文档是否出现在前 K 个检索结果里。

例如 Recall@5：

```text
正确文档是否在 top 5 里
```

### MRR

正确结果排得越靠前，分数越高。

### NDCG

考虑排序质量和相关性等级。

---

## 4.3 生成评估指标

常见维度：

- Answer Relevancy：回答是否相关
- Faithfulness：回答是否忠实于上下文
- Context Precision：检索上下文是否精准
- Context Recall：上下文是否覆盖答案

---

## 4.4 面试怎么回答“你怎么评估 RAG？”

可以这样答：

> 我会先构建一批人工标注的问题集，每个问题包含标准答案和标准来源。评估时先看检索 Recall@K，判断正确文档是否被召回；再看生成答案是否忠实于上下文。线上会记录用户 query、retrieved chunks、answer 和用户反馈，用日志持续分析和优化。

---

# Part 5：常见面试题和推荐回答

## Q1：RAG 的完整流程是什么？

回答：

> RAG 先加载文档，然后进行文本切分，用 embedding 模型把 chunk 转成向量，存入向量数据库。用户提问时，将问题也转成向量，在向量库中检索相关 chunk，把这些 chunk 作为上下文交给大模型生成回答。

---

## Q2：RAG 回答不准，你怎么排查？

回答：

> 我会先看检索结果。如果 retrieved chunks 不相关，说明是检索问题，需要优化 chunk、embedding、top_k、query rewrite、hybrid search 或 rerank。如果 retrieved chunks 是正确的但模型回答错，说明是生成问题，需要优化 prompt、加引用、加拒答逻辑或调整模型参数。

---

## Q3：向量检索和关键词检索有什么区别？

回答：

> 向量检索适合语义相似，即使字面不一样也能找到相关内容。关键词检索适合精确匹配，比如错误码、产品型号、人名、API 名称。生产中经常结合两者做 Hybrid Search。

---

## Q4：为什么需要 rerank？

回答：

> 向量检索召回速度快，但排序不一定最精准。Reranker 可以对候选文档进行更精细的相关性判断。常见做法是先召回 top 20，再 rerank 选 top 3 给 LLM。

---

## Q5：如何减少 RAG 幻觉？

回答：

> 可以从几方面做：第一，提升检索质量；第二，加相似度阈值，低于阈值拒答；第三，prompt 要求只基于上下文回答；第四，答案附引用来源；第五，记录日志并评估 faithfulness。

---

## Q6：文档更新后怎么办？

回答：

> Demo 可以全量重建，但生产中应该做增量更新。每个文档维护 doc_id，每个 chunk 维护 chunk_id。文档变化时，只删除该文档对应的 chunks，再重新切分和写入向量库。

---

## Q7：如何做权限控制？

回答：

> 在 metadata 中保存权限信息，比如部门、用户组、权限等级。检索时带 metadata filter，只召回用户有权限的文档。不能先检索敏感文档再靠 prompt 阻止模型回答，因为模型已经看到了敏感内容。

---

## Q8：Function Calling 的本质是什么？

回答：

> 模型不会真正执行函数，它只是根据用户问题生成结构化的工具调用请求，包括工具名和参数。程序解析这个请求，执行真实函数，再把结果返回给模型，让模型生成最终回答。

---

## Q9：Agent Loop 和普通 Function Calling 有什么区别？

回答：

> 普通 Function Calling 通常是一轮工具调用后直接回答。Agent Loop 会循环执行：模型判断是否调用工具，程序执行工具，把结果返回模型，模型再决定是否继续调用工具，直到不再需要工具为止。

---

## Q10：为什么模型有时候多轮调用工具，而不是一轮调用多个工具？

回答：

> 这是模型的决策策略。它可能认为工具之间有依赖，或者想先拿到一个工具结果再决定下一步。代码只要支持遍历 tool_calls，就可以处理一轮多个工具；如果模型选择串行调用，也是正常的 Agent 行为。

---

# Part 6：你当前项目可以继续升级的方向

## 优先级 1：回答加引用来源

目标：

```text
模型回答：...

参考来源：
- rag_notes.md
- profile.md
```

意义：

- 更像生产 RAG
- 方便验证答案
- 面试加分

---

## 优先级 2：相似度分数和阈值

目标：

```text
如果检索分数太低，就回答：根据当前知识库无法回答。
```

意义：

- 减少幻觉
- 更工程化

---

## 优先级 3：Rerank

目标：

```text
向量召回 top 10
rerank 选 top 3
LLM 回答
```

意义：

- 检索质量提升明显
- 面试常问

---

## 优先级 4：增量更新

目标：

```text
只更新变化的文档，不重建整个向量库。
```

意义：

- 接近生产系统
- 大规模知识库必须会

---

## 优先级 5：权限过滤

目标：

```text
metadata 加权限字段
检索时按用户权限过滤
```

意义：

- 企业级 RAG 必备

---

## 优先级 6：评估集

目标：

```text
eval_questions.json
每个问题有标准答案和标准来源
自动跑评估
```

意义：

- 证明系统真的变好了
- 面试非常加分

---

# Part 7：你需要补看的文档

建议你至少补看这些主题的官方文档或源码示例：

## Function Calling

重点看：

- tools / functions 的 schema 格式
- tool_choice
- tool_calls 返回结构
- tool role message
- tool_call_id 为什么必须传回
- 多工具调用
- Agent Loop 写法

## LangChain

重点看：

- ChatOpenAI
- Document
- Text Splitter
- VectorStore
- Retriever
- Chroma integration
- Tool calling

## Chroma

重点看：

- collection
- persist_directory
- add_documents
- delete
- metadata filter
- similarity_search
- similarity_search_with_score

## Embedding

重点看：

- embedding 维度
- normalize embeddings
- cosine similarity
- 不同 embedding 模型的差异
- 中文/英文/多语言 embedding 选择

---

# Part 8：面试时怎么介绍你的项目

你可以这样说：

> 我做了一个基于 DeepSeek + LangChain + Chroma 的本地 RAG Agent。文档从本地 md/txt 加载，使用 bge-m3 做 embedding，Chroma 做向量检索，并支持向量库持久化和文档变更自动重建。在此基础上，我把 RAG 检索封装成 search_docs 工具，结合 Function Calling 和 Agent Loop，让模型可以自主调用知识库、计算器和天气工具。后面我把代码拆成 config、models、vector_store、tools、schemas、agent 等模块，结构更接近真实工程。

然后补一句：

> 目前这个项目还是 demo 级。生产中我还会继续补充引用来源、相似度阈值、rerank、增量更新、权限过滤、评估集和日志监控。

这句话很重要，因为它能体现你知道 demo 和生产之间的差距。

---

# Part 9：你接下来最该做的 5 件事

1. 给 RAG 回答加引用来源。
2. 使用 `similarity_search_with_score` 加相似度分数。
3. 加相似度阈值，低分拒答。
4. 加 rerank 流程。
5. 做一个小型 eval 测试集。

做完这 5 个，你的项目就会从“入门 Demo”变成“有工程意识的 RAG 项目”。
