# Day 03：LangChain + Chroma 核心抽象笔记

> 今日目标：理解 LangChain 每个核心抽象解决什么问题；理解 Chroma 存了什么、怎么查、怎么更新；能把这些概念映射到自己的代码；能说出为什么这样设计，以及出问题怎么优化。

---

## 目录

1. [一句话总览](#1-一句话总览)
2. [LangChain 解决什么问题](#2-langchain-解决什么问题)
3. [LangChain 核心抽象](#3-langchain-核心抽象)
4. [核心抽象和你项目代码的对应关系](#4-核心抽象和你项目代码的对应关系)
5. [Chroma 存了什么](#5-chroma-存了什么)
6. [Chroma 怎么查](#6-chroma-怎么查)
7. [Chroma 怎么更新](#7-chroma-怎么更新)
8. [为什么这样设计](#8-为什么这样设计)
9. [出了问题怎么排查和优化](#9-出了问题怎么排查和优化)
10. [常见面试问题](#10-常见面试问题)
11. [极简背诵版](#11-极简背诵版)
12. [今天的复习任务](#12-今天的复习任务)

---

# 1. 一句话总览

LangChain 的作用：

> **LangChain 把 LLM 应用中的模型、消息、文档、切分器、Embedding、向量库、工具调用等能力封装成统一抽象，方便组合成 RAG / Agent 应用。**

Chroma 的作用：

> **Chroma 是向量数据库，用来保存文本 chunk、embedding 向量和 metadata，并支持根据用户问题做相似度检索。**

你当前项目里的关系：

```text
LangChain 负责组织流程
Chroma 负责存储和检索向量
DeepSeek 负责生成回答
bge-m3 负责文本向量化
```

---

# 2. LangChain 解决什么问题

如果不用 LangChain，你需要自己处理很多细节：

```text
不同模型 API 的调用格式
消息格式
文档对象格式
文本切分
Embedding 调用
向量库接入
工具调用格式
Agent Loop
```

LangChain 的价值是：

```text
把这些常见能力封装成统一接口
让你可以像搭积木一样组合 LLM 应用
```

例如：

```python
llm.invoke(messages)
embeddings.embed_query(query)
vector_store.similarity_search(query, k=3)
text_splitter.split_documents(docs)
```

这些接口背后屏蔽了很多底层细节。

面试表达：

> LangChain 不是模型本身，而是一个 LLM 应用开发框架。它提供 Document、ChatModel、Embeddings、TextSplitter、VectorStore、Tool 等抽象，帮助我们更快搭建 RAG 和 Agent 应用。

---

# 3. LangChain 核心抽象

## 3.1 Document

Document 是 LangChain 表示“文档”的标准对象。

它通常包含：

```text
page_content：正文内容
metadata：元数据
```

示例：

```python
Document(
    page_content="RAG 的核心流程包括...",
    metadata={"source": "rag_notes.md"},
)
```

解决的问题：

> 不同来源的文档格式不同，Document 把它们统一成一种标准格式，方便后续切分、向量化和检索。

在你项目中：

```python
# vector_store.py
Document(
    page_content=content,
    metadata={"source": filename},
)
```

---

## 3.2 ChatModel

ChatModel 是聊天模型抽象。

你项目中使用：

```python
llm = ChatOpenAI(...)
```

虽然类名叫 ChatOpenAI，但 DeepSeek 兼容 OpenAI API，所以也能用。

解决的问题：

> 不同聊天模型 API 格式不一样，ChatModel 提供统一的 `invoke()` 调用方式。

示例：

```python
response = llm.invoke(messages)
```

在你项目中：

```python
# models.py
llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)
```

---

## 3.3 Embeddings

Embeddings 是向量化模型抽象。

作用：

```text
把文本转换成向量
```

你项目中使用：

```python
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
)
```

解决的问题：

> 不同 embedding 模型调用方式不同，LangChain 用 Embeddings 抽象统一文本向量化接口。

在 RAG 里，Embedding 用于：

```text
文档 chunk 向量化
用户 query 向量化
```

---

## 3.4 TextSplitter

TextSplitter 是文本切分器。

你项目中使用：

```python
RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)
```

解决的问题：

> 原始文档可能太长，不适合直接检索或塞进模型上下文，所以需要切分成更小的 chunk。

在你项目中：

```python
# vector_store.py
def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(...)
    return text_splitter.split_documents(docs)
```

---

## 3.5 VectorStore

VectorStore 是向量数据库抽象。

你项目中使用：

```python
vector_store = Chroma.from_documents(...)
```

解决的问题：

> 不同向量数据库 API 不同，LangChain 用 VectorStore 抽象统一存储、检索、删除等接口。

常见 VectorStore：

```text
Chroma
FAISS
Milvus
Qdrant
Pinecone
Weaviate
```

常用方法：

```python
similarity_search(query, k=3)
similarity_search_with_score(query, k=3)
add_documents(docs)
delete(ids=...)
```

---

## 3.6 Retriever

Retriever 是“检索器”抽象。

VectorStore 更偏底层数据库。

Retriever 更偏“面向 RAG 的检索接口”。

通常可以这样得到：

```python
retriever = vector_store.as_retriever()
```

解决的问题：

> Retriever 把检索逻辑封装成统一接口，方便和 LangChain 的 Chain / Agent 组合。

简单理解：

```text
VectorStore：数据库能力
Retriever：检索能力封装
```

你当前项目直接使用：

```python
vector_store.similarity_search(query, k=k)
```

还没有单独使用 Retriever，这没问题。

---

## 3.7 Messages

Messages 是聊天上下文。

常见角色：

```text
system：系统指令
user：用户输入
assistant：模型回复
tool：工具结果
```

你项目中：

```python
messages = [
    {"role": "system", "content": "你是一个多工具 Agent 助手..."},
    {"role": "user", "content": user_query},
]
```

解决的问题：

> 多轮对话和工具调用都需要保留上下文，messages 就是模型看到的对话历史。

---

## 3.8 Tool / Function Calling

Tool 是模型可以调用的外部能力。

你项目里有三个工具：

```text
search_docs
calculator
get_weather
```

每个工具有两部分：

```text
真实 Python 函数：给程序执行
tool schema：给模型理解
```

在你项目中：

```python
# tools.py
AVAILABLE_FUNCTIONS = {
    "search_docs": search_docs,
    "calculator": calculator,
    "get_weather": get_weather,
}
```

```python
# schemas.py
TOOLS = [search_docs_schema, calculator_schema, get_weather_schema]
```

---

# 4. 核心抽象和你项目代码的对应关系

| LangChain / 工程概念 | 解决什么问题 | 你项目中的位置 |
|---|---|---|
| Document | 统一文档格式 | `vector_store.py -> load_local_docs()` |
| ChatModel | 统一聊天模型调用 | `models.py -> llm` |
| Embeddings | 文本向量化 | `models.py -> embeddings` |
| TextSplitter | 文本切分 | `vector_store.py -> split_docs()` |
| VectorStore | 存储和检索向量 | `vector_store.py -> Chroma` |
| Messages | 保存对话上下文 | `agent.py -> messages` |
| Tool Function | 程序真正执行工具 | `tools.py` |
| Tool Schema | 给模型看的工具说明 | `schemas.py` |
| Agent Loop | 多轮工具调用循环 | `agent.py -> run_agent()` |
| Config | 统一配置路径和参数 | `config.py` |

---

# 5. Chroma 存了什么

Chroma 是向量数据库。

它通常存四类东西：

```text
1. 文本内容，也就是 chunk 的 page_content
2. 文本对应的 embedding 向量
3. metadata，例如 source 文件名
4. id，用来唯一标识一条记录
```

你项目中，Chroma 存的是：

```text
chunk 文本
bge-m3 生成的向量
metadata={"source": filename}
```

创建代码：

```python
vector_store = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DB_DIR,
)
```

这段代码做了几件事：

```text
接收切分后的 chunks
调用 embeddings 把 chunk 转成向量
把文本、向量、metadata 存入 Chroma
保存到 chroma_db 文件夹
```

面试表达：

> Chroma 存的不只是向量，还会保存原始文本和 metadata。检索时返回的 Document 里就包含 page_content 和 metadata。

---

# 6. Chroma 怎么查

你项目使用：

```python
retrieved_docs = vector_store.similarity_search(query, k=k)
```

这个过程内部发生：

```text
用户 query
  ↓
用同一个 embedding 模型转成 query vector
  ↓
在 Chroma 中和已有 chunk vectors 计算相似度
  ↓
返回最相似的 top-k 个 Document
```

返回结果类似：

```python
[
    Document(
        page_content="RAG 的核心流程包括...",
        metadata={"source": "rag_notes.md"},
    )
]
```

如果想看分数，可以用：

```python
vector_store.similarity_search_with_score(query, k=3)
```

它会返回：

```python
[(Document(...), score), (Document(...), score)]
```

注意：不同向量库或距离算法中，score 含义可能不同，有的是越小越相似，有的是越大越相似，需要看具体实现。

---

# 7. Chroma 怎么更新

## 7.1 你当前项目的更新方式

你现在的方案是：

```text
检测 docs 文件状态是否变化
  ↓
如果变化，删除整个 chroma_db
  ↓
重新切分全部文档
  ↓
重新 embedding
  ↓
重新创建 Chroma
```

相关代码：

```python
if need_rebuild and os.path.exists(CHROMA_DB_DIR):
    shutil.rmtree(CHROMA_DB_DIR)
```

优点：

```text
简单
适合 Demo
不容易出现脏数据
```

缺点：

```text
文档多了以后很慢
每次变化都要全量 embedding
成本高
不适合生产
```

---

## 7.2 生产中的更新方式

生产更常见的是增量更新。

基本思路：

```text
每个文档有 doc_id
每个 chunk 有 chunk_id
metadata 中记录 source / doc_id / chunk_id
某个文档变化时，只删除这个文档对应的 chunks
再重新插入这个文档的新 chunks
```

可能用到的方法：

```python
vector_store.add_documents(new_docs)
vector_store.delete(ids=[...])
```

或者按 metadata 删除，具体取决于向量库支持能力。

面试表达：

> Demo 可以全量重建，但生产中应该做增量更新。文档变更时，只更新对应 doc_id 的 chunks，避免全量重建带来的成本和延迟。

---

# 8. 为什么这样设计

## 8.1 为什么要用 Document

因为来源不同的文档格式不统一。

Document 统一成：

```text
正文 + metadata
```

这样后面切分、向量化、检索都可以用同一套接口。

---

## 8.2 为什么要用 TextSplitter

因为原文太长，直接 embedding 整篇文档会导致：

```text
检索粒度太粗
相关内容被大量无关内容稀释
上下文太长，成本高
```

切成 chunk 后，检索更精准。

---

## 8.3 为什么要用 Embedding

因为用户问题和文档不一定字面完全一样。

Embedding 可以做语义匹配。

例如：

```text
用户问：RAG 怎么工作？
文档写：RAG 的核心流程包括文档加载、切分、向量化、检索、生成。
```

关键词不完全一致，但语义相关。

---

## 8.4 为什么要用 Vector Store

因为向量需要高效存储和检索。

如果文档很多，不能每次都暴力遍历所有向量。

向量数据库可以：

```text
持久化存储
高效相似度检索
保存 metadata
支持增删改查
```

---

## 8.5 为什么要用 metadata

metadata 可以保存来源和业务信息。

例如：

```text
source 文件名
page 页码
doc_id 文档 ID
chunk_id 片段 ID
department 部门
permission 权限
created_at 创建时间
```

用途：

```text
答案引用来源
权限过滤
增量更新
问题排查
日志分析
```

---

# 9. 出了问题怎么排查和优化

## 9.1 文档没被检索到

可能原因：

```text
docs 没加载成功
文件为空
文件后缀不是 .txt / .md
chroma_db 没重建
metadata 或签名逻辑有问题
```

排查：

```python
print(len(docs))
print(len(splits))
```

---

## 9.2 检索结果不相关

可能原因：

```text
chunk_size 不合适
query 太模糊
embedding 模型效果不好
top_k 太小或太大
文档内容本身太少
```

基础优化：

```text
调整 chunk_size / chunk_overlap
调整 k
换 embedding 模型
打印 retrieved_docs 看检索结果
```

---

## 9.3 Chroma 里还是旧内容

可能原因：

```text
chroma_db 已存在，程序直接加载旧库
文件签名没有变化
更新逻辑没有触发重建
```

解决：

```text
删除 chroma_db 重新运行
检查 docs_signature.txt
检查 get_docs_signature()
```

---

## 9.4 模型回答错，但检索结果是对的

说明问题可能在生成阶段。

优化：

```text
改 prompt
要求只基于上下文回答
要求不知道就说不知道
降低 temperature
加引用来源
```

---

## 9.5 模型回答错，而且检索结果也错

说明问题主要在检索阶段。

优化：

```text
检查文档切分
检查 embedding
调整 top_k
增加 query rewrite
后续可加 rerank
```

面试表达：

> RAG 出问题时，我会先看 retrieved chunks。如果检索不对，优先优化切分、embedding、top-k；如果检索对但回答错，再优化 prompt 和生成策略。

---

# 10. 常见面试问题

## Q1：LangChain 的 Document 是什么？

回答：

> Document 是 LangChain 对文档的统一抽象，主要包含 page_content 和 metadata。page_content 存正文，metadata 存来源、页码、权限等信息。这样不同来源的文档可以统一进入后续的切分、向量化和检索流程。

---

## Q2：Embedding 和 ChatModel 有什么区别？

回答：

> Embedding 模型负责把文本转成向量，用于语义检索；ChatModel 负责生成自然语言回答。Embedding 不聊天，ChatModel 不负责向量检索。

---

## Q3：TextSplitter 解决什么问题？

回答：

> TextSplitter 把长文档切成更小的 chunk，解决文档过长、检索粒度太粗、上下文成本高的问题。chunk 是 RAG 的基本检索单位。

---

## Q4：VectorStore 和 Retriever 有什么区别？

回答：

> VectorStore 更像底层向量数据库，负责存储和相似度查询；Retriever 是面向 RAG 的检索接口封装，通常可以由 vector_store.as_retriever() 得到。简单说，VectorStore 是存储层，Retriever 是检索接口层。

---

## Q5：Chroma 里面到底存了什么？

回答：

> Chroma 通常存文本 chunk、chunk 对应的 embedding 向量、metadata 和 id。检索时会根据 query embedding 找到相似向量，再返回对应文本和 metadata。

---

## Q6：persist_directory 是干什么的？

回答：

> persist_directory 用来指定 Chroma 数据库保存到本地哪个目录。设置后，向量库可以持久化保存，下次程序启动时可以直接加载，不需要重新 embedding 全部文档。

---

## Q7：Chroma 怎么更新？

回答：

> Demo 里可以检测文档变化后删除整个 chroma_db 全量重建。生产中更推荐增量更新，给文档和 chunk 维护 id，文档变化时只删除该文档对应 chunks，再插入新 chunks。

---

## Q8：如果检索不准，你怎么排查？

回答：

> 我会先打印 retrieved docs，看是否召回了正确内容。如果没召回，检查文档加载、chunk 切分、embedding、top-k 和向量库是否更新。如果召回正确但回答错误，再优化 prompt 和生成策略。

---

# 11. 极简背诵版

1. **LangChain 是 LLM 应用开发框架，不是模型本身。**
2. **Document = page_content + metadata。**
3. **ChatModel 负责生成回答，Embeddings 负责文本向量化。**
4. **TextSplitter 负责把长文档切成 chunk。**
5. **VectorStore 负责存储和检索向量。**
6. **Retriever 是面向 RAG 的检索接口封装。**
7. **Chroma 存 chunk 文本、embedding 向量、metadata 和 id。**
8. **similarity_search 会把 query 向量化，再找最相似的 top-k chunks。**
9. **persist_directory 用来持久化保存 Chroma。**
10. **Demo 可以全量重建，生产应该做增量更新。**
11. **metadata 可用于来源引用、权限过滤、增量更新和排查问题。**
12. **RAG 出错先看检索结果，再判断是检索问题还是生成问题。**

---

# 12. 今天的复习任务

请用自己的话回答下面 6 个问题：

1. LangChain 的 Document 解决了什么问题？
2. ChatModel 和 Embeddings 的区别是什么？
3. TextSplitter 为什么重要？
4. Chroma 里面存了什么？
5. similarity_search 的内部流程是什么？
6. 如果 Chroma 检索结果不准，你怎么排查？
