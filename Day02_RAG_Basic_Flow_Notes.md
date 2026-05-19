# Day 02：RAG 基础全流程笔记

> 今日目标：能清楚讲明白 **RAG 从文档到最终回答的完整流程**。本篇只整理基础流程，不展开 RAG 优化，优化内容后面单独补。

---

## 目录

1. [一句话总览](#1-一句话总览)
2. [RAG 完整流程](#2-rag-完整流程)
3. [文档加载](#3-文档加载)
4. [Chunk 文本切分](#4-chunk-文本切分)
5. [Embedding 向量化](#5-embedding-向量化)
6. [Vector Store 向量数据库](#6-vector-store-向量数据库)
7. [Similarity Search 相似度检索](#7-similarity-search-相似度检索)
8. [Top-K](#8-top-k)
9. [Prompt 拼接](#9-prompt-拼接)
10. [最终生成](#10-最终生成)
11. [面试讲法](#11-面试讲法)
12. [极简背诵版](#12-极简背诵版)

---

# 1. 一句话总览

RAG 的全称是 Retrieval-Augmented Generation，中文叫 **检索增强生成**。

一句话解释：

> RAG 是先从外部知识库检索和用户问题相关的资料，再把资料作为上下文交给大模型生成回答。

它解决的问题：

```text
大模型不知道私有知识
大模型知识可能过期
大模型容易胡编
不想为了知识更新频繁微调模型
```

---

# 2. RAG 完整流程

RAG 基础流程可以分成两个阶段：

## 2.1 离线阶段：构建知识库

```text
文档加载
  ↓
文本切分 chunk
  ↓
Embedding 向量化
  ↓
存入 Vector Store
```

## 2.2 在线阶段：用户提问并生成回答

```text
用户提问
  ↓
问题 Embedding 向量化
  ↓
Similarity Search 相似度检索
  ↓
取 Top-K 相关文档片段
  ↓
Prompt 拼接：问题 + 检索上下文
  ↓
LLM 最终生成回答
```

你当前项目里的对应关系：

| RAG 环节 | 你项目里的实现 |
|---|---|
| 文档加载 | `load_local_docs()` |
| 文本切分 | `split_docs()` |
| Embedding | `HuggingFaceEmbeddings("BAAI/bge-m3")` |
| Vector Store | `Chroma` |
| 相似度检索 | `vector_store.similarity_search()` |
| Top-K | `k=2` / `k=3` |
| Prompt / Agent | `run_agent()` 中模型基于工具结果回答 |
| 最终生成 | DeepSeek `llm.invoke()` |

---

# 3. 文档加载

文档加载是 RAG 的第一步。

作用：

> 把外部知识源读取进程序，变成统一的 Document 对象。

常见知识源：

```text
.txt
.md
.pdf
Word
Excel
网页
数据库
Notion / 飞书 / Confluence
```

你当前项目先处理最简单的：

```text
.md
.txt
```

代码示例：

```python
def load_local_docs(docs_dir: str):
    docs = []

    for filename in os.listdir(docs_dir):
        if not filename.endswith((".txt", ".md")):
            continue

        file_path = os.path.join(docs_dir, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        docs.append(
            Document(
                page_content=content,
                metadata={"source": filename},
            )
        )

    return docs
```

Document 通常包含两部分：

```text
page_content：文档正文
metadata：文档元数据，比如来源文件名、页码、权限、时间等
```

面试表达：

> 文档加载的作用是把不同来源的资料统一转换成 Document 格式，方便后续切分、向量化和检索。

---

# 4. Chunk 文本切分

Chunk 是文档切分后的文本片段。

为什么要切分？

```text
原始文档可能很长，不能全部放进模型上下文
检索整篇文档粒度太粗，不够精准
模型只需要和问题相关的片段
```

示例：

原始文档：

```text
RAG 的核心流程包括：加载文档、文本切分、Embedding 向量化、存入向量数据库、根据用户问题检索相关片段、把片段作为上下文交给大模型回答。
```

切分后可能变成：

```text
chunk 1：RAG 的核心流程包括：加载文档、文本切分、Embedding 向量化...
chunk 2：存入向量数据库、根据用户问题检索相关片段、把片段作为上下文...
```

你项目里的代码：

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)

splits = text_splitter.split_documents(docs)
```

## 4.1 chunk_size

`chunk_size` 表示每个 chunk 大概多长。

例如：

```python
chunk_size=100
```

表示每个片段大约 100 个字符。

## 4.2 chunk_overlap

`chunk_overlap` 表示相邻 chunk 之间重叠多少字符。

例如：

```python
chunk_overlap=20
```

为什么需要 overlap？

> 防止一句话或一个概念刚好被切断，保留上下文连续性。

面试表达：

> Chunk 是 RAG 的检索粒度。切得太大，检索不精准；切得太小，语义可能不完整。所以需要根据文档类型调整 chunk_size 和 chunk_overlap。

---

# 5. Embedding 向量化

Embedding 是把文本转换成向量。

向量可以理解为一串数字：

```python
[0.012, -0.234, 0.756, ...]
```

为什么要向量化？

> 因为计算机可以通过向量距离判断两个文本的语义是否相似。

例如：

```text
问题：RAG 的流程是什么？
文档：RAG 包括文档加载、文本切分、向量化、检索和生成。
```

虽然字面不完全一样，但语义相近，embedding 后向量距离会比较近。

你项目里的 embedding 模型：

```python
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
```

这里：

```text
BAAI/bge-m3：embedding 模型
CPU：用 CPU 运行
normalize_embeddings：归一化向量，让相似度计算更稳定
```

面试表达：

> Embedding 的作用是把文本映射到向量空间，语义相近的文本在向量空间里距离更近。RAG 通过 embedding 实现语义检索。

---

# 6. Vector Store 向量数据库

Vector Store 用来存储和检索向量。

它一般存：

```text
文本 chunk
chunk 对应的 embedding 向量
metadata
id
```

常见向量数据库：

```text
Chroma
FAISS
Milvus
Qdrant
Pinecone
Weaviate
Elasticsearch / OpenSearch
```

你项目里用的是 Chroma：

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
拿到切分后的 chunks
用 embedding 模型把 chunk 转成向量
把 chunk + 向量 + metadata 存入 Chroma
持久化保存到 chroma_db 文件夹
```

面试表达：

> Vector Store 用来保存文本片段及其向量表示，并支持根据用户问题向量进行相似度检索。

---

# 7. Similarity Search 相似度检索

Similarity Search 是 RAG 在线阶段的核心步骤。

流程：

```text
用户输入问题
  ↓
把问题转成 query embedding
  ↓
和向量数据库中的 chunk embedding 计算相似度
  ↓
返回最相似的文档片段
```

你项目里的代码：

```python
retrieved_docs = vector_store.similarity_search(query, k=k)
```

这行代码内部做了：

```text
query 向量化
向量相似度计算
返回 top-k 相关 chunks
```

返回结果通常是 Document 列表：

```python
[
    Document(page_content="RAG 的核心流程包括...", metadata={"source": "rag_notes.md"}),
    Document(page_content="小明熟悉 RAG 和 Chroma...", metadata={"source": "profile.md"}),
]
```

面试表达：

> Similarity Search 是把用户问题和知识库 chunk 都放到同一个向量空间中，通过相似度计算找到最相关的文本片段。

---

# 8. Top-K

Top-K 表示返回最相关的前 K 个结果。

例如：

```python
similarity_search(query, k=3)
```

表示返回最相关的 3 个 chunk。

## 8.1 K 太小的问题

```text
可能漏掉答案
上下文不够完整
```

## 8.2 K 太大的问题

```text
无关信息变多
prompt 变长
token 成本增加
模型可能被干扰
```

所以 K 需要根据任务调试。

Demo 中常用：

```text
k=2
k=3
```

面试表达：

> Top-K 控制召回多少个相关片段。K 太小可能漏召回，K 太大又会引入噪音和成本，所以需要根据数据和效果调参。

---

# 9. Prompt 拼接

检索到相关 chunk 后，需要把它们拼进 prompt，让模型基于这些资料回答。

基础 prompt 结构：

```text
请你只根据下面的上下文回答用户问题。
如果上下文里没有答案，请说不知道。

上下文：
{context}

用户问题：
{query}
```

代码示例：

```python
context = "\n\n".join([doc.page_content for doc in retrieved_docs])

prompt = f"""
请你只根据下面的上下文回答用户问题。
如果上下文里没有答案，请说不知道。

上下文：
{context}

用户问题：
{query}
"""
```

在你当前项目里，RAG 被封装成了工具：

```python
def search_docs(query: str, k: int = 2):
    retrieved_docs = vector_store.similarity_search(query, k=k)
    ...
```

模型调用 `search_docs` 后，工具结果会作为 tool message 放回 `messages`，模型再根据工具结果回答。

面试表达：

> Prompt 拼接就是把检索到的相关上下文和用户问题组合起来，引导模型只基于上下文生成答案。

---

# 10. 最终生成

最终生成就是 LLM 根据：

```text
用户问题
检索到的上下文
系统指令
```

生成自然语言回答。

你项目里使用 DeepSeek：

```python
answer = llm.invoke(prompt)
```

或者在 Agent Loop 中：

```python
response = llm.invoke(
    messages,
    tools=TOOLS,
    tool_choice="auto",
)
```

RAG 的关键原则：

> 检索质量决定回答上限，生成模型决定表达质量。

如果检索结果错了，模型很难回答正确。

如果检索结果对了，但 prompt 写得不好，模型也可能答偏。

面试表达：

> 最终生成阶段是让 LLM 基于检索上下文回答用户问题。RAG 的关键是让模型有依据地回答，而不是完全依赖模型自身记忆。

---

# 11. 面试讲法

如果面试官问：**你讲一下 RAG 的完整流程。**

可以这样回答：

> RAG 分为离线构建知识库和在线问答两个阶段。离线阶段先加载文档，把文档切分成 chunk，然后用 embedding 模型把 chunk 转成向量，存入向量数据库。在线阶段用户提问后，系统把问题也转成向量，在向量库中做 similarity search，取 top-k 个相关 chunk，把这些 chunk 拼成上下文，与用户问题一起组成 prompt，最后交给大模型生成回答。

如果想结合你的项目说：

> 我的项目里文档从 docs 文件夹中的 md/txt 加载，使用 RecursiveCharacterTextSplitter 切分文本，用 BAAI/bge-m3 做 embedding，Chroma 做向量数据库。用户提问时通过 similarity_search 检索 top-k 文档片段，然后让 DeepSeek 基于检索结果生成回答。

---

# 12. 极简背诵版

1. **RAG = 检索增强生成。**
2. **RAG 不是直接让模型回答，而是先查知识库，再基于资料回答。**
3. **离线阶段：文档加载 → chunk → embedding → 存入 vector store。**
4. **在线阶段：用户问题 → query embedding → similarity search → top-k chunks → prompt 拼接 → LLM 生成。**
5. **文档加载是把外部资料转成统一的 Document。**
6. **Chunk 是检索粒度，太大不精准，太小语义不完整。**
7. **Embedding 是把文本转成向量，用于语义相似度计算。**
8. **Vector Store 保存 chunk、向量和 metadata。**
9. **Similarity Search 是根据问题向量找最相似的文档片段。**
10. **Top-K 控制返回多少个相关片段。**
11. **Prompt 拼接是把检索上下文和用户问题组合后交给模型。**
12. **最终生成是 LLM 基于上下文输出自然语言答案。**

---

# 13. 今天的复习任务

请用自己的话回答下面 5 个问题：

1. RAG 的完整流程是什么？
2. 为什么要把文档切成 chunk？
3. Embedding 在 RAG 里起什么作用？
4. Vector Store 存的是什么？
5. Top-K 太大或太小分别有什么问题？
