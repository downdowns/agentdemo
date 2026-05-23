# LangChain RAG：文档加载、切分、Embedding、检索与生成

> 本文是学习型知识库文档，基于 LangChain 官方 RAG、Text Splitters 文档和本项目实践整理，用于企业知识库 Agent 的 RAG 检索实验。

## 1. RAG 是什么

RAG 是 Retrieval-Augmented Generation，即检索增强生成。它的核心思想是：模型回答问题前，先从外部知识库检索相关内容，再把检索结果作为上下文交给大模型生成答案。

RAG 解决的是大模型应用中的几个常见问题：

1. **知识更新问题**：模型训练后的新知识无法自动进入参数，RAG 可以通过更新知识库补充新信息。
2. **私有知识问题**：企业内部文档、制度、接口说明、项目文档通常不在通用模型训练集中。
3. **可追溯问题**：检索结果可以带有来源信息，便于回答时引用出处。
4. **减少幻觉**：模型不完全依赖参数记忆，而是基于检索上下文回答。

在企业知识库问答中，RAG 通常比直接微调更适合快速落地，因为文档更新成本低、来源可追踪、实现链路清晰。

## 2. RAG 的基本流程

一个标准 RAG 流程通常分为离线构建和在线查询两部分。

离线构建阶段：

```text
原始文档 -> 文档加载 -> 文本切分 -> embedding 向量化 -> 存入向量数据库
```

在线查询阶段：

```text
用户问题 -> 问题向量化 -> 向量相似度检索 -> 取 top-k chunk -> 拼接 prompt -> LLM 生成答案
```

本项目中，`vector_store.py` 负责离线构建，`search_docs` 工具负责在线检索，Agent 通过 Function Calling 决定什么时候调用检索工具。

## 3. 为什么要切分文档

企业文档通常很长，不能直接把整篇文档塞进 embedding 或 prompt。文档切分的目的包括：

1. **适配模型上下文窗口**：大模型输入长度有限，检索结果必须控制大小。
2. **提高检索粒度**：用户问题通常只对应文档中的一小段，chunk 级检索比整篇文档检索更精准。
3. **降低噪声**：整篇文档可能包含大量无关内容，切分后能只取相关片段。
4. **提升引用能力**：chunk 可以带有 source、chunk_id、标题路径等 metadata，便于定位答案依据。

如果 chunk 太大，检索结果容易包含噪声；如果 chunk 太小，语义上下文可能不完整。因此 chunk_size 和 chunk_overlap 是 RAG 优化中非常重要的参数。

## 4. chunk_size 和 chunk_overlap

`chunk_size` 表示每个文本片段的目标长度。长度可以按字符、token 或自定义函数计算，具体取决于 splitter 配置。

`chunk_overlap` 表示相邻 chunk 之间保留多少重叠内容。它的作用是避免关键信息刚好被切分边界截断。

例如：

```text
chunk_001: A B C D E
chunk_002: D E F G H
```

这里 D E 是 overlap。这样即使问题需要 D、E、F 的上下文，第二个 chunk 仍然包含足够信息。

常见调参经验：

- FAQ、短知识点：chunk 可以小一些；
- 技术文档、制度文档：chunk 可以中等；
- 长报告、论文：可能需要层级切分或按标题切分；
- overlap 太大会增加重复内容和向量库体积；
- overlap 太小可能导致上下文断裂。

## 5. RecursiveCharacterTextSplitter 的思路

LangChain 的 `RecursiveCharacterTextSplitter` 会按一组分隔符递归切分文本。通常会优先尝试较大的结构边界，例如段落，再尝试句子、空格，最后才按字符切分。

这种方式比粗暴固定长度切分更适合普通 Markdown 或文本文件，因为它尽量保留自然段落结构。对于中文文档，实际项目中还可以根据标点、标题层级和 Markdown 结构进一步优化。

本项目最初使用较小的 chunk_size 方便学习，后续为了更真实的 RAG 评估，可以把 chunk_size 调整到 300-800 字符，并为每个 chunk 增加 `chunk_id`、`source`、`chunk_index` 等 metadata。

## 6. Embedding 是什么

Embedding 是把文本转换成向量的过程。向量是一个高维数字数组，用来表达文本语义。语义相近的文本，在向量空间中的距离通常更近。

例如：

```text
“LangGraph checkpoint 有什么作用”
```

和包含“checkpoint 保存图状态快照”的 chunk，语义上接近，因此向量相似度应该较高。

本项目使用 BGE-M3 作为 embedding 模型，把文档 chunk 和用户问题都转换成向量，然后用 Chroma 做相似度检索。

## 7. top-k 检索

检索时通常会返回最相似的前 k 个 chunk。k 太小可能漏掉关键信息，k 太大可能引入噪声并增加 prompt 长度。

例如：

```text
top_k = 3
```

表示取最相关的 3 个 chunk。后续做 chunk-level Recall@k 时，我们会检查 expected_chunk_ids 是否出现在前 k 个实际检索结果中。

## 8. RAG 的常见优化方向

RAG 优化通常不只调模型，而是围绕检索链路展开：

1. **文档清洗**：去掉目录、页眉页脚、重复内容和无意义符号。
2. **切分优化**：调整 chunk_size、chunk_overlap，或改用 Markdown 标题切分、语义切分。
3. **metadata 设计**：为 chunk 增加 source、chunk_id、section、title_path 等字段。
4. **检索策略**：调整 top_k，结合向量检索和关键词检索做 hybrid search。
5. **rerank**：先召回较多候选，再用 reranker 对结果重新排序。
6. **query rewrite**：对用户问题进行改写、扩展或拆解，提高召回率。
7. **评估体系**：使用 Recall@k、MRR、Hit@k、答案正确性、引用一致性等指标。

## 9. 本项目下一步实践

本项目准备从 source-level 评估升级为 chunk-level 评估。核心改造包括：

1. 切分后为每个 chunk 增加唯一 `chunk_id`；
2. Chroma metadata 中保存 `source`、`chunk_id`、`chunk_index`；
3. `search_docs` 返回 top-k chunk 的 source、chunk_id 和 content；
4. `questions.json` 中标注 `expected_chunk_ids`；
5. `run_eval.py` 计算 chunk-level Recall@k。

这样，评估不再只判断“是否命中文档”，而是判断“是否命中具体相关片段”。这更接近真实 RAG 项目。

## 参考来源

- LangChain RAG 文档：https://docs.langchain.com/oss/python/langchain/rag
- LangChain Text Splitters 文档：https://docs.langchain.com/oss/python/integrations/splitters/index
- LangChain RecursiveCharacterTextSplitter 文档：https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter
