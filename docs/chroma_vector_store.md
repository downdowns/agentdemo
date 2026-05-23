# Chroma 向量数据库：Collection、Document、Metadata 与 Query

> 本文是学习型知识库文档，基于 Chroma 官方文档和本项目实践整理，用于企业知识库 RAG 检索实验。

## 1. Chroma 在 RAG 中的角色

Chroma 是一个常用的向量数据库。它在 RAG 系统中的作用是保存文本 chunk 的向量表示，并在用户查询时进行相似度检索。

在本项目中，Chroma 存储三类核心信息：

1. **文本内容**：也就是 chunk 的正文，后续会作为上下文交给大模型。
2. **向量表示**：由 embedding 模型根据文本内容生成，用于相似度搜索。
3. **元数据 metadata**：例如 source、chunk_id、chunk_index，用于追踪来源和做评估。

Chroma 不是大模型，也不负责生成答案。它负责“找资料”。大模型负责“基于资料组织回答”。

## 2. Collection 是什么

Collection 可以理解为 Chroma 中的一组向量数据集合。一个项目可以有一个或多个 collection。

例如：

```text
collection_name = "enterprise_rag_docs"
```

可以表示企业知识库文档集合。后续如果系统支持多个知识库，也可以为不同业务线建立不同 collection，例如：

```text
hr_policy_docs
tech_docs
finance_docs
```

本项目当前使用一个 collection 保存 docs 目录下的所有文档 chunk。

## 3. Document 和 Metadata

在 LangChain 中，一个文档片段通常是 `Document` 对象：

```python
Document(
    page_content="checkpoint 是图状态的快照...",
    metadata={
        "source": "langgraph_persistence.md",
        "chunk_id": "langgraph_persistence.md::chunk_003",
        "chunk_index": 3
    }
)
```

其中：

- `page_content` 是实际参与 embedding 和问答的文本内容；
- `metadata` 是不会直接作为主要语义内容的结构化信息；
- `source` 用来表示来自哪篇文档；
- `chunk_id` 用来唯一标识某个切片；
- `chunk_index` 用来表示它是该文档的第几个 chunk。

良好的 metadata 设计对 RAG 很重要。没有 metadata，系统即使检索到了正确内容，也很难解释“答案来自哪里”，也无法做 chunk-level Recall@k。

## 4. Query 是怎么工作的

当用户提出问题时，RAG 系统会把问题也转换成向量，然后到 Chroma 中查找相似向量。

简化流程如下：

```text
用户问题 -> embedding -> query vector -> Chroma 相似度检索 -> top-k chunks
```

如果使用 LangChain 的 Chroma 集成，通常可以通过 `similarity_search` 之类的方法得到检索结果。每个结果包含 page_content 和 metadata。

检索结果可能长这样：

```json
{
  "content": "Checkpoint 是图状态的快照...",
  "source": "langgraph_persistence.md",
  "chunk_id": "langgraph_persistence.md::chunk_003"
}
```

这些结果随后会被格式化成 prompt 上下文，交给大模型生成最终答案。

## 5. 持久化目录

Chroma 可以把向量数据库持久化到本地目录。这样下次启动项目时，如果 docs 没有变化，就不用重新 embedding 所有文档。

本项目通过 docs 签名判断是否需要重建向量库：

1. 读取 docs 目录下文件名、修改时间和文件大小；
2. 与上一次保存的签名比较；
3. 如果发生变化，删除旧 Chroma 数据库并重建；
4. 如果没有变化，直接加载已有向量库。

这个机制适合开发阶段，能避免每次启动都重复构建。

## 6. 为什么要保存 chunk_id

如果只保存 source，那么系统只能知道命中了哪篇文档：

```text
langgraph_persistence.md
```

但真实 RAG 评估更关心命中了哪一个具体片段：

```text
langgraph_persistence.md::chunk_003
```

chunk_id 的作用包括：

1. 支持 chunk-level Recall@k；
2. 支持更精确的引用定位；
3. 支持分析哪些 chunk 经常被命中；
4. 支持后续做 rerank 前后对比；
5. 支持调试 chunk_size 和 chunk_overlap 是否合理。

因此，下一步我们会修改 `vector_store.py`，在切分后给每个 chunk 增加 chunk_id。

## 7. Chroma 与 RAG 优化的关系

Chroma 负责向量检索，但检索质量不只由 Chroma 决定。影响检索质量的因素包括：

- embedding 模型是否适合中文和技术文档；
- chunk 切分是否保留完整语义；
- query 是否表达清楚；
- top_k 设置是否合适；
- metadata 是否能支持过滤和评估；
- 是否使用 rerank 对候选结果重排。

因此，如果 RAG 回答不好，不能简单认为是大模型问题，也可能是文档切分、向量化、检索排序或上下文拼接的问题。

## 8. 面试回答模板

如果面试官问“Chroma 里存了什么”，可以回答：

> Chroma 存的是文档 chunk 的向量、文本内容和 metadata。向量用于相似度检索，文本内容用于拼接到 prompt 给模型回答，metadata 用于记录 source、chunk_id 等来源信息。后续做 RAG 评估时，可以根据 chunk_id 计算 chunk-level Recall@k，根据 source 做来源追踪。

## 参考来源

- Chroma Collection 参考：https://docs.trychroma.com/reference/python/collection
- Chroma Query and Get 文档：https://docs.trychroma.com/docs/querying-collections/query-and-get
