"""Agent 可以调用的真实 Python 工具函数。

这个文件里的函数是真正会被 Python 执行的工具。

注意区分：
- tools.py：给程序执行的真实函数
- schemas.py：给模型看的工具说明书

Function Calling 的完整链路是：
1. 模型根据 schemas.py 里的工具说明生成 tool_calls
2. Agent 根据 tool_calls 里的 name 找到 AVAILABLE_FUNCTIONS 中的函数
3. Python 执行这里定义的真实函数
4. 工具结果再作为 ToolMessage 返回给模型
"""

import re
from typing import Any

from sentence_transformers import CrossEncoder

from vector_store import vector_store

# Rerank 开关：
# - True：先从 Chroma 多召回候选，再用词项重叠 baseline 重排。
# - False：直接返回 Chroma 原始向量检索 top-k。
#
# 注意：这个开关主要用于评估实验，方便对比“纯向量检索”和“向量检索 + rerank”。
# 不把它暴露到 schemas.py，是因为它不应该由模型决定，而应该由系统/实验配置决定。
USE_RERANK = True

# Rerank 模式：
# - "keyword"：使用当前的词项重叠 rerank baseline
# - "cross_encoder"：使用正式 CrossEncoder reranker 模型
#
# 先默认使用 keyword，避免一开始就加载大模型导致运行变慢。
RERANK_MODE = "keyword"

# 正式 reranker 模型：
# CrossEncoder 会同时看 query 和 chunk，然后输出相关性分数。
# 这里选择 BAAI/bge-reranker-base，和当前 BGE-M3 embedding 风格比较一致。
CROSS_ENCODER_RERANKER_MODEL = "BAAI/bge-reranker-base"

# 用于缓存已经加载好的 reranker 模型。
# 初始值是 None，表示模型还没有加载。
_cross_encoder_reranker: CrossEncoder | None = None

# Query Rewrite 开关：
# - True：先把用户 query 改写成更适合检索的 query，再去 Chroma 检索。
# - False：直接使用原始 query 检索。
#
# 注意：这是一个 rule-based baseline，不是 LLM rewrite。
# 这个 baseline 在当前项目里多次实验后效果不佳，
# 所以默认关闭；需要做实验时再手动打开。
USE_QUERY_REWRITE = False

def _should_rewrite_query(query: str) -> bool:
    """判断一个 query 是否值得做 rewrite。

    保守策略：
    - 太短的 query，通常信息不够，适合补词
    - 带指代/口语化词的 query，通常容易歧义，适合补词
    - 已经很完整的长 query，尽量不要改，避免 query drift
    """
    normalized_query = query.strip()

    if not normalized_query:
        return False

    # 很短的 query，通常需要补充上下文。
    if len(normalized_query) <= 12:
        return True

    # 口语化 / 指代性很强的问题，更适合做轻量 rewrite。
    ambiguous_markers = [
        "它",
        "这个",
        "那个",
        "这",
        "那",
        "什么",
        "怎么",
        "为什么",
        "如何",
        "有啥",
        "作用",
        "关系",
        "区别",
    ]
    if any(marker in normalized_query for marker in ambiguous_markers):
        return True

    return False

def rewrite_query_for_search(query: str) -> str:
    """把用户 query 改写成更适合 RAG 检索的 query。

    当前是 rule-based baseline，不调用额外 LLM。

    思路：
    - 保留原始 query
    - 根据 query 中出现的关键词，补充相关术语
    - 让向量检索和 rerank 更容易命中目标 chunk

    这不是完美改写，只是一个可解释 baseline。
    """
    normalized_query = query.strip()

    # 长 query 本身已经足够清楚时，不做 rewrite，
    # 避免把原始意图“冲淡”。
    if not _should_rewrite_query(normalized_query):
        return normalized_query

    # 先保留原始 query。
    rewritten_terms = [normalized_query]

    # 转小写主要是为了匹配英文关键词时更稳。
    lower_query = normalized_query.lower()

    # RAG / 文档切分相关。
    if any(keyword in lower_query for keyword in ["rag", "切分", "chunk", "chunk_size", "chunk_overlap"]):
        rewritten_terms.append(
            "RAG 文档切分 chunk_size chunk_overlap"
        )

    # Chroma / 向量库相关。
    if any(keyword in lower_query for keyword in ["chroma", "向量库", "向量数据库", "metadata", "chunk_id"]):
        rewritten_terms.append(
            "Chroma 向量数据库 metadata chunk_id"
        )

    # LangGraph checkpoint / thread_id / 记忆相关。
    if any(keyword in lower_query for keyword in ["checkpoint", "thread_id", "inmemorysaver", "记忆", "会话", "隔离"]):
        rewritten_terms.append(
            "LangGraph checkpoint thread_id InMemorySaver"
        )

    # LangGraph reducer / state 相关。
    if any(keyword in lower_query for keyword in ["reducer", "messagesstate", "add_messages", "tool_calls", "sources"]):
        rewritten_terms.append(
            "LangGraph reducer add_messages tool_calls sources"
        )

    # FastAPI 请求校验相关。
    if any(keyword in lower_query for keyword in ["fastapi", "pydantic", "校验", "http", "接口"]):
        rewritten_terms.append(
            "FastAPI Pydantic HTTPException 400"
        )

    # RAG 优化相关。
    if any(keyword in lower_query for keyword in ["优化", "不准确", "召回", "rerank", "rewrite", "hybrid"]):
        rewritten_terms.append(
            "RAG 优化 Recall@k MRR rerank query rewrite"
        )

    # 用空格拼起来，作为最终检索 query。
    return " ".join(rewritten_terms)

def get_cross_encoder_reranker() -> CrossEncoder:
    """获取 CrossEncoder reranker 模型。

    这里使用懒加载：
    - 第一次调用时才真正加载模型
    - 加载后保存到全局变量 _cross_encoder_reranker
    - 后续再次调用时直接复用，避免重复加载
    """
    global _cross_encoder_reranker

    if _cross_encoder_reranker is None:
        _cross_encoder_reranker = CrossEncoder(CROSS_ENCODER_RERANKER_MODEL)

    return _cross_encoder_reranker

def _extract_rerank_terms(text: str) -> set[str]:
    """把 query 或 chunk 内容拆成适合做简单 rerank 的词项集合。

    这个 baseline 不依赖额外模型，只做最小的词项匹配：
    - 英文 / 数字 / 下划线：按完整词保留
    - 中文：按连续 2 字符窗口切分，提升中文命中率
    """
    terms: set[str] = set()

    # 先把文本里的英文、数字和连续中文块提出来。
    for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text.lower()):
        # 英文词直接保留。
        if re.fullmatch(r"[A-Za-z0-9_]+", token):
            if len(token) > 1:
                terms.add(token)
            continue

        # 单字中文也保留，避免极短关键词丢失。
        if len(token) == 1:
            terms.add(token)
            continue

        # 中文块拆成 bigram，例如“为什么要切分” -> “为什么/么要/要切/切分”
        for i in range(len(token) - 1):
            terms.add(token[i : i + 2])

    return terms


def _score_doc_for_query(query: str, doc) -> int:
    """给一个 chunk 计算最简单的相关性分数。

    分数越高，说明 query 和 chunk 的词项重叠越多。
    这是一个可解释的 rerank baseline，不是正式 reranker 模型。
    """
    query_terms = _extract_rerank_terms(query)
    doc_text = " ".join(
        [
            doc.page_content,
            str(doc.metadata.get("source", "")),
            str(doc.metadata.get("chunk_id", "")),
        ]
    ).lower()

    return sum(1 for term in query_terms if term and term in doc_text)

def _score_doc_with_cross_encoder(query: str, doc) -> float:
    """使用正式 CrossEncoder reranker 给一个 chunk 打分。

    CrossEncoder 会把 query 和 chunk 内容作为一对文本输入，
    直接输出一个相关性分数。

    分数越高，表示该 chunk 越适合回答当前 query。
    """
    reranker = get_cross_encoder_reranker()

    # doc.page_content 是当前 chunk 的正文内容。
    # CrossEncoder 的输入格式是 list[list[str]]：
    # [
    #     [query, chunk_text]
    # ]
    score = reranker.predict([[query, doc.page_content]])[0]

    # predict 返回的可能是 numpy.float32，这里转成 Python float，
    # 方便后续 json 序列化和日志打印。
    return float(score)

def search_docs(query: str, k: int = 2) -> list[dict[str, Any]]:
    """
    搜索本地知识库。

    参数：
    - query：用户问题或关键词
    - k：返回最相关的前 k 个文档片段

    返回：
    - source：文档来源文件名
    - chunk_id：文档片段唯一 ID，用于 chunk-level 检索评估
    - chunk_index：当前片段在原文档中的序号
    - content：检索到的文档内容
    - rewritten_query：实际用于检索的 query（如果开启 query rewrite）
    """
    # 如果开启 query rewrite，就先把 query 改写成更适合检索的版本。
    # 这一步的目标是让“短问题 / 口语化问题 / 省略主语的问题”更容易命中相关 chunk。
    rewritten_query = rewrite_query_for_search(query) if USE_QUERY_REWRITE else query

    # 第一步：根据 USE_RERANK 决定召回数量。
    # - 不开 rerank：只召回 k 个，直接作为最终结果。
    # - 开 rerank：先多召回一些候选，再做第二阶段重排。
    candidate_k = max(k * 3, 5) if USE_RERANK else k

    # similarity_search 会做两件事：
    # 1. 使用 vector_store 绑定的 embedding 模型把 query 转成向量
    # 2. 在 Chroma 里找出向量距离最接近的 candidate_k 个文档片段
    #
    # 当前这是“召回阶段”；真实业务里可以继续扩展为：
    # - top-k 调优
    # - hybrid search
    # - rerank
    # - metadata filter
    retrieved_docs = vector_store.similarity_search(rewritten_query, k=candidate_k)

    if USE_RERANK:
        # 第二步：对候选 chunk 做 rerank。
        #
        # RERANK_MODE 用来控制具体使用哪种 rerank 方法：
        # - keyword：词项重叠 baseline，速度快、可解释，但不懂深层语义
        # - cross_encoder：正式 reranker 模型，效果通常更好，但会更慢
        scored_docs = []

        for idx, doc in enumerate(retrieved_docs):
            if RERANK_MODE == "keyword":
                score = _score_doc_for_query(rewritten_query, doc)
            elif RERANK_MODE == "cross_encoder":
                score = _score_doc_with_cross_encoder(rewritten_query, doc)
            else:
                raise ValueError(f"Unknown RERANK_MODE: {RERANK_MODE}")

            scored_docs.append((score, idx, doc))

        # 分数高的排前面；分数相同则保持原始召回顺序。
        scored_docs.sort(key=lambda item: (-item[0], item[1]))

        # 第三步：只返回重排后的前 k 个 chunk。
        final_docs = [doc for _, _, doc in scored_docs[:k]]
    else:
        # 关闭 rerank 时，直接使用 Chroma 的原始向量检索顺序。
        final_docs = retrieved_docs[:k]

    # 把 LangChain Document 对象转换成普通 dict，方便 json.dumps 和工具返回。
    results: list[dict[str, Any]] = []
    for doc in final_docs:
        results.append(
            {
                "source": doc.metadata.get("source", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                "chunk_index": doc.metadata.get("chunk_index", -1),
                "content": doc.page_content,
                "rewritten_query": rewritten_query if USE_QUERY_REWRITE else None,
                # 方便调试和观察 rerank 是否起作用。
                # 关闭 rerank 时记为 None，表示没有经过第二阶段重排。
                "rerank_score": (
                    _score_doc_for_query(rewritten_query, doc)
                    if USE_RERANK and RERANK_MODE == "keyword"
                    else _score_doc_with_cross_encoder(rewritten_query, doc)
                    if USE_RERANK and RERANK_MODE == "cross_encoder"
                    else None
                ),
                "rerank_mode": RERANK_MODE if USE_RERANK else None,
            }
        )

    # 返回检索结果列表。
    # 这里不直接返回 LangChain Document 对象，是因为：
    # - Document 不能很好地 JSON 序列化
    # - Agent 工具结果需要能 json.dumps
    # - FastAPI 返回也更适合普通 dict/list
    return results


def calculator(operation: str, a: float, b: float) -> dict[str, Any]:
    """
    执行基础数学运算。

    operation 支持：
    - add：加法
    - subtract：减法
    - multiply：乘法
    - divide：除法
    """
    # 用字典把 operation 字符串映射到具体的计算函数。
    # 这样比写很多 if/elif 更清晰，也方便扩展新运算。
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero",
    }

    # 如果模型传入了不支持的 operation，就返回错误信息。
    # 注意：工具参数来自模型输出，不能假设一定合法。
    if operation not in operations:
        return {"error": "Unknown operation"}

    try:
        # 根据 operation 找到对应函数，并传入 a、b 执行。
        return {"result": operations[operation](a, b)}
    except Exception as e:
        # 如果计算过程出现异常，把错误信息返回给模型。
        return {"error": str(e)}


def get_weather(city: str) -> dict[str, str]:
    """
    查询城市天气。

    注意：这里是 demo 工具，返回的是模拟天气数据，不是真实天气 API。
    以后如果你想接真实天气，可以在这个函数里调用真实天气接口。
    """
    # 模拟天气数据：key 是城市名，value 是天气信息。
    mock_weather = {
        "上海": {"weather": "晴天", "temperature": "26°C"},
        "北京": {"weather": "多云", "temperature": "22°C"},
        "广州": {"weather": "小雨", "temperature": "28°C"},
        "深圳": {"weather": "阴天", "temperature": "27°C"},
    }

    # 根据城市名取天气。
    # 如果城市不在 mock_weather 里，就返回“未知”。
    weather_info = mock_weather.get(
        city,
        {"weather": "未知", "temperature": "未知"},
    )

    # 返回结构化结果，方便模型理解和组织回答。
    # 工具返回尽量使用 dict，而不是自然语言字符串；
    # 这样模型更容易提取字段，日志和评估也更方便。
    return {
        "city": city,
        "weather": weather_info["weather"],
        "temperature": weather_info["temperature"],
        "note": "这是 demo 模拟天气，不是真实实时天气。",
    }


# 给 Python 程序看的工具映射。
#
# 模型只会返回类似：
# {"name": "calculator", "args": {...}}
#
# Agent 代码会用 function_name 到 AVAILABLE_FUNCTIONS 里查找真实函数：
# AVAILABLE_FUNCTIONS[function_name](**function_args)
#
# 如果以后要新增工具，通常需要同时改两个地方：
# 1. tools.py：新增真实函数，并加入 AVAILABLE_FUNCTIONS
# 2. schemas.py：新增工具 schema，让模型知道这个工具存在
AVAILABLE_FUNCTIONS = {
    "search_docs": search_docs,
    "calculator": calculator,
    "get_weather": get_weather,
}
