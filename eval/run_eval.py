"""最小 Agent / RAG 评估脚本。

当前评估目标：
    1. 检查 Agent 是否调用了期望工具。
    2. 检查 RAG 检索是否命中了期望文档来源。
    3. 检查 RAG 检索是否命中了期望 chunk。
    4. 支持对比“纯向量检索”和“向量检索 + rerank baseline”。
    5. 支持对比“原始 query”和“query rewrite 后的 query”。
    6. 检查答案命中的关键点是否能被 retrieved chunks 支撑。

评估流程：
    1. 读取 eval/questions.json
    2. 根据 use_rerank 设置 tools.USE_RERANK
    3. 对每个问题调用 run_agent()
    4. 从 Agent 返回的 tool_calls 中提取 actual_tools
    5. 从 Agent 返回结果中提取 actual_sources
    6. 从 search_docs 工具结果中提取 actual_chunk_ids
    7. 从 search_docs 工具结果中提取 retrieved_context
    8. 计算 Tool Call Pass Rate、Source Hit Rate、Chunk Recall@1、Chunk Recall@3、MRR@3、Answer Point Hit Rate、Citation Faithfulness Rate

运行方式：
    # 默认：开启 rerank baseline，关闭 query rewrite，跑一遍评估
    python eval/run_eval.py

    # 关闭 rerank，跑纯向量检索 baseline
    python eval/run_eval.py --no-rerank

    # 分别跑 no-rerank / rerank，并输出对比
    python eval/run_eval.py --compare-rerank

    # 关闭 query rewrite，跑原始 query baseline（默认也是这个）
    python eval/run_eval.py --no-query-rewrite

    # 分别跑 no-rewrite / rewrite，并输出对比
    python eval/run_eval.py --compare-query-rewrite

    # 对比 keyword rerank 和 cross_encoder rerank 的逐题差异
    python eval/run_eval.py --compare-rerank-modes
"""

import argparse
import json
import sys
from pathlib import Path

# run_eval.py 在 eval 目录中，agent.py 在项目根目录。
# 为了支持 `python eval/run_eval.py` 这种运行方式，需要把项目根目录加入 sys.path。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import tools
from agent import run_agent
from prompts import get_system_prompt

# 当前 run_eval.py 所在文件夹 / questions.json。
# 这样不管从哪个目录启动脚本，都能正确找到评估问题集。
QUESTIONS_PATH = Path(__file__).parent / "questions.json"


def load_questions() -> list[dict]:
    """读取评估问题集。

    questions.json 的每一条数据包含：
    - id：问题编号
    - question：用户问题
    - expected_tools：期望 Agent 调用的工具列表
    - expected_sources：期望 RAG 检索命中的文档来源列表
    - expected_chunk_ids：期望 RAG 检索命中的 chunk_id 列表
    """
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        # json.load 会把 JSON 数组解析为 Python list。
        return json.load(f)


def extract_tool_names(tool_calls: list[dict]) -> list[str]:
    """从 Agent 返回的 tool_calls 中提取工具名。

    Agent 返回的 tool_calls 结构大致是：
    [
        {"name": "calculator", "args": {...}, "result": {...}}
    ]

    评估工具调用准确性时，我们只关心 name 字段。
    """
    tool_names = []

    for tool_call in tool_calls:
        # 用 get 比 tool_call["name"] 更安全：
        # 如果某条记录缺少 name 字段，不会直接抛 KeyError。
        name = tool_call.get("name")
        if name:
            tool_names.append(name)

    return tool_names


def extract_sources(result: dict) -> list[str]:
    """从 Agent 返回结果中提取 sources。

    Agent 返回的 sources 表示 RAG 检索命中的文档来源。
    如果 result 中没有 sources，就返回空列表。
    """
    sources = result.get("sources", [])

    if sources is None:
        return []

    return sources


def extract_rewritten_queries_from_tool_calls(tool_calls: list[dict]) -> list[str]:
    """从 search_docs 工具结果中提取 rewritten_query。

    search_docs 现在会返回 rewritten_query，方便我们观察：
    - 原始 query
    - 改写后的 query

    这个字段主要用于调试和评估，不影响主流程。
    """
    rewritten_queries: list[str] = []

    for tool_call in tool_calls:
        if tool_call.get("name") != "search_docs":
            continue

        search_results = tool_call.get("result", [])

        if not isinstance(search_results, list):
            continue

        for item in search_results:
            if not isinstance(item, dict):
                continue

            rewritten_query = item.get("rewritten_query")
            if rewritten_query and rewritten_query not in rewritten_queries:
                rewritten_queries.append(rewritten_query)

    return rewritten_queries

def extract_answer(result: dict) -> str:
    """从 Agent 返回结果中提取最终回答 answer。

    后续做答案质量评估时，会检查这个 answer 是否覆盖 expected_answer_points。

    使用 get 是为了防御异常情况：
    - 正常情况下 result["answer"] 是字符串
    - 如果某次返回没有 answer，就返回空字符串，避免程序报 KeyError
    """
    answer = result.get("answer", "")

    if answer is None:
        return ""

    return str(answer)

def hit_answer_point(answer: str, point: dict) -> bool:
    """判断最终回答是否命中某个答案关键点。

    point 结构示例：
    {
        "point": "文档切分可以适配模型上下文窗口限制",
        "keywords": ["上下文窗口", "输入长度", "长度有限"]
    }

    当前采用最简单的关键词命中策略：
    只要 answer 中包含 keywords 里的任意一个关键词，
    就认为这个 point 被命中。

    这是一个 baseline，不是完美评估方法。
    """
    keywords = point.get("keywords", [])

    for keyword in keywords:
        if keyword and keyword in answer:
            return True

    return False

def calculate_answer_point_score(
    answer: str,
    expected_answer_points: list[dict],
) -> tuple[float | None, list[str], list[str]]:
    """计算答案关键点命中率。

    返回：
    - score：
        命中的关键点数量 / 总关键点数量
        如果 expected_answer_points 为空，返回 None，表示不参与答案质量评估。
    - hit_points：
        被命中的 point 名称列表。
    - missed_points：
        没有命中的 point 名称列表。
    """
    if not expected_answer_points:
        return None, [], []

    hit_points = []
    missed_points = []

    for point in expected_answer_points:
        point_name = point.get("point", "")

        if hit_answer_point(answer, point):
            hit_points.append(point_name)
        else:
            missed_points.append(point_name)

    score = len(hit_points) / len(expected_answer_points)

    return score, hit_points, missed_points


def extract_retrieved_context_from_tool_calls(tool_calls: list[dict]) -> str:
    """从 search_docs 工具结果中提取 retrieved context。

    retrieved_context 指的是：
    - 本轮 Agent 调用 search_docs 后实际返回给模型的文档片段正文；
    - 也就是模型生成 RAG 答案时理论上能看到的外部知识依据。

    后续做 Citation Faithfulness 时，会检查：
    答案里命中的关键点，是否也能在 retrieved_context 中找到关键词依据。
    """
    context_parts: list[str] = []

    for tool_call in tool_calls:
        if tool_call.get("name") != "search_docs":
            continue

        search_results = tool_call.get("result", [])

        if not isinstance(search_results, list):
            continue

        for item in search_results:
            if not isinstance(item, dict):
                continue

            content = item.get("content")
            if content:
                context_parts.append(str(content))

    return "\n\n".join(context_parts)


def hit_context_point(retrieved_context: str, point: dict) -> bool:
    """判断 retrieved_context 是否支撑某个答案关键点。

    当前仍然采用最小关键词 baseline：
    - 如果 retrieved_context 中包含该 point 的任意一个 keyword，
      就认为检索上下文可以支撑这个 point。

    这不是严格的语义蕴含判断，但足够作为第一版可解释 baseline。
    后续可以升级为：
    - LLM-as-Judge
    - NLI / entailment 模型
    - 人工抽检
    """
    keywords = point.get("keywords", [])

    for keyword in keywords:
        if keyword and keyword in retrieved_context:
            return True

    return False


def calculate_citation_faithfulness_score(
    answer: str,
    retrieved_context: str,
    expected_answer_points: list[dict],
) -> tuple[float | None, list[str], list[str]]:
    """计算 Citation Faithfulness 分数。

    Citation Faithfulness 关注：
    “答案中已经命中的关键点，是否能被检索到的 context 支撑？”

    计算方式：
    1. 先判断 answer 命中了哪些 expected_answer_points；
    2. 对这些“答案已声称/已覆盖”的 point，再检查 retrieved_context 是否包含对应关键词；
    3. 分数 = supported_points / answer_hit_points。

    如果 answer 没有命中任何 point，则返回 None：
    - 这种情况更应该由 Answer Point Hit Rate 负责；
    - Faithfulness 不评价“没有说出来的内容”是否有依据。
    """
    if not expected_answer_points:
        return None, [], []

    supported_points = []
    unsupported_points = []

    for point in expected_answer_points:
        point_name = point.get("point", "")

        # 只检查“答案里已经命中的关键点”。
        # 如果答案没提这个 point，就不算 citation 不忠实，
        # 而是由 Answer Point Hit Rate 记录为 missed point。
        if not hit_answer_point(answer, point):
            continue

        if hit_context_point(retrieved_context, point):
            supported_points.append(point_name)
        else:
            unsupported_points.append(point_name)

    total_claimed_points = len(supported_points) + len(unsupported_points)

    if total_claimed_points == 0:
        return None, [], []

    score = len(supported_points) / total_claimed_points

    return score, supported_points, unsupported_points


def extract_chunk_ids_from_tool_calls(tool_calls: list[dict]) -> list[str]:
    """从 search_docs 的工具调用结果中提取 chunk_id 列表。

    Agent 的 tool_calls 中会记录每个工具的执行结果。
    对于 search_docs，result 通常是 list[dict]，每个 dict 表示一个检索 chunk。
    """
    chunk_ids: list[str] = []

    for tool_call in tool_calls:
        # 只关心知识库检索工具；calculator / get_weather 不会返回 chunk_id。
        if tool_call.get("name") != "search_docs":
            continue

        search_results = tool_call.get("result", [])

        # 工具异常时 result 可能不是 list，所以这里防御性跳过。
        if not isinstance(search_results, list):
            continue

        for item in search_results:
            if not isinstance(item, dict):
                continue

            chunk_id = item.get("chunk_id")
            if chunk_id:
                chunk_ids.append(chunk_id)

    return chunk_ids


def is_pass(expected_tools: list[str], actual_tools: list[str]) -> bool:
    """判断期望工具是否都被实际调用。

    当前采用“宽松判断”：
    只要 expected_tools 中的工具都出现在 actual_tools 中，就算通过。
    """
    for tool in expected_tools:
        if tool not in actual_tools:
            return False

    return True


def is_source_hit(expected_sources: list[str], actual_sources: list[str]) -> bool:
    """判断期望来源是否都被实际检索命中。

    如果 expected_sources 为空，说明该问题不需要 RAG 检索，
    直接认为 source 评估通过。
    """
    if not expected_sources:
        return True

    for source in expected_sources:
        if source not in actual_sources:
            return False

    return True


def calculate_recall_at_k(
    expected_chunk_ids: list[str],
    actual_chunk_ids: list[str],
    k: int,
) -> float | None:
    """计算 chunk-level Recall@k。

    Recall@k 公式：
        前 k 个实际检索 chunk 中命中的 expected_chunk_ids 数量
        /
        expected_chunk_ids 总数量

    如果 expected_chunk_ids 为空，说明当前样本不是 RAG 检索题，
    不参与 Recall@k 统计，返回 None。
    """
    if not expected_chunk_ids:
        return None

    top_k_chunk_ids = actual_chunk_ids[:k]

    hit_count = 0
    for chunk_id in expected_chunk_ids:
        if chunk_id in top_k_chunk_ids:
            hit_count += 1

    return hit_count / len(expected_chunk_ids)

def calculate_mrr_at_k(
    expected_chunk_ids: list[str],
    actual_chunk_ids: list[str],
    k: int,
) -> float | None:
    """计算 MRR@k。

    MRR = Mean Reciprocal Rank。

    它关注的是：
    第一个正确 chunk 在前 k 个检索结果中排第几名。

    例如：
    - 正确 chunk 排第 1：得分 1 / 1 = 1.0
    - 正确 chunk 排第 2：得分 1 / 2 = 0.5
    - 正确 chunk 排第 3：得分 1 / 3 = 0.333
    - 前 k 个都没命中：得分 0

    如果 expected_chunk_ids 为空，说明不是 RAG 题，不参与统计。
    """
    if not expected_chunk_ids:
        return None

    top_k_chunk_ids = actual_chunk_ids[:k]

    for rank, chunk_id in enumerate(top_k_chunk_ids, start=1):
        if chunk_id in expected_chunk_ids:
            return 1 / rank

    return 0.0


def set_eval_modes(
    *,
    use_rerank: bool,
    rerank_mode: str,
    use_query_rewrite: bool,
) -> None:
    """统一设置 tools.py 里的实验开关。

    use_rerank 控制是否开启 rerank。
    rerank_mode 控制使用哪一种 rerank：
    - keyword：词项重叠 baseline
    - cross_encoder：正式 CrossEncoder reranker
    use_query_rewrite 控制是否开启 query rewrite。
    """
    tools.USE_RERANK = use_rerank
    tools.RERANK_MODE = rerank_mode
    tools.USE_QUERY_REWRITE = use_query_rewrite


def evaluate_questions(
    questions: list[dict],
    *,
    use_rerank: bool,
    rerank_mode: str,
    use_query_rewrite: bool,
    prompt_version: str,
    label: str,
    verbose: bool = True,
) -> dict:
    """执行一次完整评估，并返回汇总指标。

    参数：
    - questions：评估集。
    - use_rerank：是否开启 tools.py 中的 rerank baseline。
    - rerank_mode：rerank 模式，keyword 或 cross_encoder。
    - use_query_rewrite：是否开启 query rewrite。
    - label：本次实验名称，例如 baseline_no_rerank / rerank_baseline。
    - verbose：是否打印每道题的详细过程。做两组实验对比时可以关掉，
      避免终端输出过长，只保留最终 summary 和差异分析。
    """
    # 关键：这里设置 tools.py 的全局开关。
    # search_docs() 执行时会读取这些开关，决定是否走 rerank / query rewrite 流程。
    set_eval_modes(
        use_rerank=use_rerank,
        rerank_mode=rerank_mode,
        use_query_rewrite=use_query_rewrite,
    )

    system_prompt = get_system_prompt(prompt_version)

    total = len(questions)
    passed_count = 0
    source_hit_count = 0
    recall_at_1_sum = 0.0
    recall_at_3_sum = 0.0
    rag_eval_count = 0
    low_recall_cases = []
    mrr_at_3_sum = 0.0
    low_mrr_cases = []
    # 所有参与答案质量评估的问题分数总和。
    answer_point_score_sum = 0.0
    # 有 expected_answer_points 的问题数量。
    answer_point_eval_count = 0
    # 记录低分答案，方便后面分析。
    low_answer_quality_cases = []
    # Citation Faithfulness 分数总和。
    citation_faithfulness_score_sum = 0.0
    # 参与 Citation Faithfulness 评估的问题数量。
    citation_faithfulness_eval_count = 0
    # 记录低 citation faithfulness 样本，方便定位“答案有关键点，但检索上下文不支撑”的情况。
    low_citation_faithfulness_cases = []
    # 记录每一道 RAG 题的检索结果，后面用于做 rerank mode 差异分析。
    case_results = []

    print("\n" + "=" * 80)
    print(f"开始评估：{label}")
    print("USE_RERANK:", tools.USE_RERANK)
    print("RERANK_MODE:", tools.RERANK_MODE)
    print("USE_QUERY_REWRITE:", tools.USE_QUERY_REWRITE)
    print("PROMPT_VERSION:", prompt_version)
    print(f"共加载 {total} 条评估问题")

    for item in questions:
        if verbose:
            print("\n---------------------------------")
            print("id:", item["id"])
            print("question:", item["question"])
            print("expected_tools:", item["expected_tools"])

        # 调用 Agent，拿到结构化返回结果。
        result = run_agent(
            item["question"],
            system_prompt=system_prompt,
        )

        # 提取 Agent 最终回答。
        # 这一步暂时只打印，不打分。
        # 后面 40.3 会用 actual_answer 去匹配 expected_answer_points。
        actual_answer = extract_answer(result)
        if verbose:
            print("actual_answer:", actual_answer)

        expected_answer_points = item.get("expected_answer_points", [])

        answer_point_score, hit_points, missed_points = calculate_answer_point_score(
            actual_answer,
            expected_answer_points,
        )

        if verbose:
            print("Answer Point Score:", answer_point_score)
            print("Hit Points:", hit_points)
            print("Missed Points:", missed_points)

        if answer_point_score is not None:
            answer_point_score_sum += answer_point_score
            answer_point_eval_count += 1

        # 提取实际工具调用。
        actual_tools = extract_tool_names(result["tool_calls"])
        if verbose:
            print("actual tool_calls:", actual_tools)

        # 提取实际命中的文档来源。
        actual_sources = extract_sources(result)
        if verbose:
            print("expected_sources:", item.get("expected_sources", []))
            print("actual_sources:", actual_sources)

        actual_rewritten_queries = extract_rewritten_queries_from_tool_calls(result["tool_calls"])
        if verbose:
            print("actual_rewritten_queries:", actual_rewritten_queries)

        # 提取实际命中的 chunk_id。
        actual_chunk_ids = extract_chunk_ids_from_tool_calls(result["tool_calls"])
        if verbose:
            print("actual_chunk_ids:", actual_chunk_ids)

        # 提取本轮 search_docs 返回的正文内容。
        # Citation Faithfulness 会用它判断答案关键点是否有检索依据。
        retrieved_context = extract_retrieved_context_from_tool_calls(result["tool_calls"])
        if verbose:
            print("retrieved_context_length:", len(retrieved_context))

        citation_faithfulness_score, supported_points, unsupported_points = (
            calculate_citation_faithfulness_score(
                actual_answer,
                retrieved_context,
                expected_answer_points,
            )
        )

        if verbose:
            print("Citation Faithfulness Score:", citation_faithfulness_score)
            print("Supported Points:", supported_points)
            print("Unsupported Points:", unsupported_points)

        if citation_faithfulness_score is not None:
            citation_faithfulness_score_sum += citation_faithfulness_score
            citation_faithfulness_eval_count += 1

        expected_chunk_ids = item.get("expected_chunk_ids", [])

        recall_at_1 = calculate_recall_at_k(
            expected_chunk_ids,
            actual_chunk_ids,
            k=1,
        )
        recall_at_3 = calculate_recall_at_k(
            expected_chunk_ids,
            actual_chunk_ids,
            k=3,
        )
        mrr_at_3 = calculate_mrr_at_k(
            expected_chunk_ids,
            actual_chunk_ids,
            k=3,
        )

        if verbose:
            print("expected_chunk_ids:", expected_chunk_ids)
            print("Recall@1:", recall_at_1)
            print("Recall@3:", recall_at_3)
            print("MRR@3:", mrr_at_3)

        # 只有真正的 RAG 题才参与 Recall@k 统计。
        if recall_at_1 is not None and recall_at_3 is not None and mrr_at_3 is not None:
            recall_at_1_sum += recall_at_1
            recall_at_3_sum += recall_at_3
            mrr_at_3_sum += mrr_at_3
            rag_eval_count += 1
            case_results.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "expected_chunk_ids": expected_chunk_ids,
                    "actual_top3_chunk_ids": actual_chunk_ids[:3],
                    "recall_at_1": recall_at_1,
                    "recall_at_3": recall_at_3,
                    "mrr_at_3": mrr_at_3,
                    "answer_point_score": answer_point_score,
                    "citation_faithfulness_score": citation_faithfulness_score,
                    "actual_rewritten_queries": actual_rewritten_queries,
                }
            )

            if recall_at_1 < 1.0:
                low_recall_cases.append(
                    {
                        "id": item["id"],
                        "question": item["question"],
                        "expected_chunk_ids": expected_chunk_ids,
                        "actual_top3_chunk_ids": actual_chunk_ids[:3],
                        "recall_at_1": recall_at_1,
                        "recall_at_3": recall_at_3,
                        "actual_rewritten_queries": actual_rewritten_queries,
                    }
                )
            
            if mrr_at_3 < 1.0:
                low_mrr_cases.append(
                    {
                        "id": item["id"],
                        "question": item["question"],
                        "expected_chunk_ids": expected_chunk_ids,
                        "actual_top3_chunk_ids": actual_chunk_ids[:3],
                        "mrr_at_3": mrr_at_3,
                    }
                )

        # 如果答案关键点命中率较低，也记录下来。
        # 这里放在 Recall@k 计算之后，是为了同时记录 actual_top3_chunk_ids 和 recall_at_3。
        # 这样后面分析低质量答案时，可以判断到底是“检索没找对”，还是“资料找到了但模型没答全”。
        if answer_point_score is not None and answer_point_score < 0.75:
            low_answer_quality_cases.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "answer": actual_answer,
                    "answer_point_score": answer_point_score,
                    "hit_points": hit_points,
                    "missed_points": missed_points,
                    "actual_top3_chunk_ids": actual_chunk_ids[:3],
                    "recall_at_3": recall_at_3,
                    "actual_rewritten_queries": actual_rewritten_queries,
                }
            )

        # 如果答案关键点命中了，但 retrieved_context 不能支撑，也记录下来。
        # 这类样本通常表示：
        # - 模型可能使用了参数知识；
        # - 或者检索结果没有提供足够依据；
        # - 或者关键词 baseline 太粗，需要后续升级为 LLM-as-Judge。
        if (
            citation_faithfulness_score is not None
            and citation_faithfulness_score < 1.0
        ):
            low_citation_faithfulness_cases.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "citation_faithfulness_score": citation_faithfulness_score,
                    "supported_points": supported_points,
                    "unsupported_points": unsupported_points,
                    "answer": actual_answer,
                    "actual_top3_chunk_ids": actual_chunk_ids[:3],
                    "retrieved_context_length": len(retrieved_context),
                }
            )

        source_hit = is_source_hit(item.get("expected_sources", []), actual_sources)
        if verbose:
            print("source_hit:", source_hit)

        if source_hit:
            source_hit_count += 1

        passed = is_pass(item["expected_tools"], actual_tools)
        if verbose:
            print("passed:", passed)

        if passed:
            passed_count += 1

    pass_rate = passed_count / total if total > 0 else 0
    source_hit_rate = source_hit_count / total if total > 0 else 0
    avg_recall_at_1 = recall_at_1_sum / rag_eval_count if rag_eval_count > 0 else 0
    avg_recall_at_3 = recall_at_3_sum / rag_eval_count if rag_eval_count > 0 else 0
    avg_mrr_at_3 = mrr_at_3_sum / rag_eval_count if rag_eval_count > 0 else 0

    avg_answer_point_score = (
        answer_point_score_sum / answer_point_eval_count
        if answer_point_eval_count > 0
        else 0
    )
    avg_citation_faithfulness_score = (
        citation_faithfulness_score_sum / citation_faithfulness_eval_count
        if citation_faithfulness_eval_count > 0
        else 0
    )

    metrics = {
        "label": label,
        "use_rerank": use_rerank,
        "rerank_mode": rerank_mode,
        "prompt_version": prompt_version,
        "total": total,
        "tool_call_pass_count": passed_count,
        "tool_call_pass_rate": pass_rate,
        "source_hit_count": source_hit_count,
        "source_hit_rate": source_hit_rate,
        "rag_eval_count": rag_eval_count,
        "chunk_recall_at_1": avg_recall_at_1,
        "chunk_recall_at_3": avg_recall_at_3,
        "mrr_at_3": avg_mrr_at_3,
        "low_recall_cases": low_recall_cases,
        "low_mrr_cases": low_mrr_cases,
        "answer_point_eval_count": answer_point_eval_count,
        "answer_point_hit_rate": avg_answer_point_score,
        "low_answer_quality_cases": low_answer_quality_cases,
        "citation_faithfulness_eval_count": citation_faithfulness_eval_count,
        "citation_faithfulness_rate": avg_citation_faithfulness_score,
        "low_citation_faithfulness_cases": low_citation_faithfulness_cases,
        "case_results": case_results,
    }

    print_eval_summary(metrics)
    return metrics


def print_eval_summary(metrics: dict) -> None:
    """打印单次评估汇总结果。"""
    print("\n---------------------------------")
    print("评估完成：", metrics["label"])
    print("USE_RERANK：", metrics["use_rerank"])
    print("RERANK_MODE：", metrics.get("rerank_mode"))
    print("PROMPT_VERSION：", metrics.get("prompt_version"))
    print("总题数：", metrics["total"])
    print("工具调用通过数 Tool Call Pass Count：", metrics["tool_call_pass_count"])
    print(f"工具调用通过率 Tool Call Pass Rate：{metrics['tool_call_pass_rate']:.2%}")
    print("来源命中数 Source Hit Count：", metrics["source_hit_count"])
    print(f"来源命中率 Source Hit Rate：{metrics['source_hit_rate']:.2%}")
    print("RAG 评估题数：", metrics["rag_eval_count"])
    print(f"Chunk Recall@1：{metrics['chunk_recall_at_1']:.2%}")
    print(f"Chunk Recall@3：{metrics['chunk_recall_at_3']:.2%}")
    print(f"MRR@3：{metrics['mrr_at_3']:.2%}")
    print("答案质量评估题数：", metrics["answer_point_eval_count"])
    print(f"Answer Point Hit Rate：{metrics['answer_point_hit_rate']:.2%}")
    print("Citation Faithfulness 评估题数：", metrics["citation_faithfulness_eval_count"])
    print(f"Citation Faithfulness Rate：{metrics['citation_faithfulness_rate']:.2%}")

    print("\n低 Recall@1 样本：")
    if not metrics["low_recall_cases"]:
        print("无")
    else:
        for case in metrics["low_recall_cases"]:
            print("---------------------------------")
            print("id:", case["id"])
            print("question:", case["question"])
            print("expected_chunk_ids:", case["expected_chunk_ids"])
            print("actual_top3_chunk_ids:", case["actual_top3_chunk_ids"])
            print("actual_rewritten_queries:", case.get("actual_rewritten_queries", []))
            print("Recall@1:", case["recall_at_1"])
            print("Recall@3:", case["recall_at_3"])

    print("\n低 MRR@3 样本：")
    if not metrics["low_mrr_cases"]:
        print("无")
    else:
        for case in metrics["low_mrr_cases"]:
            print("---------------------------------")
            print("id:", case["id"])
            print("question:", case["question"])
            print("expected_chunk_ids:", case["expected_chunk_ids"])
            print("actual_top3_chunk_ids:", case["actual_top3_chunk_ids"])
            print("MRR@3:", case["mrr_at_3"])
    
    print("\n低答案质量样本：")
    if not metrics["low_answer_quality_cases"]:
        print("无")
    else:
        for case in metrics["low_answer_quality_cases"]:
            print("---------------------------------")
            print("id:", case["id"])
            print("question:", case["question"])
            print("Answer Point Score:", case["answer_point_score"])
            print("Hit Points:", case["hit_points"])
            print("Missed Points:", case["missed_points"])
            print("actual_top3_chunk_ids:", case["actual_top3_chunk_ids"])
            print("actual_rewritten_queries:", case.get("actual_rewritten_queries", []))
            print("Recall@3:", case["recall_at_3"])
            print("answer:", case["answer"])

    print("\n低 Citation Faithfulness 样本：")
    if not metrics["low_citation_faithfulness_cases"]:
        print("无")
    else:
        for case in metrics["low_citation_faithfulness_cases"]:
            print("---------------------------------")
            print("id:", case["id"])
            print("question:", case["question"])
            print("Citation Faithfulness Score:", case["citation_faithfulness_score"])
            print("Supported Points:", case["supported_points"])
            print("Unsupported Points:", case["unsupported_points"])
            print("actual_top3_chunk_ids:", case["actual_top3_chunk_ids"])
            print("retrieved_context_length:", case["retrieved_context_length"])
            print("answer:", case["answer"])


def print_compare_summary(baseline: dict, rerank: dict) -> None:
    """打印 no-rerank 与 rerank 的指标对比。"""
    recall_1_delta = rerank["chunk_recall_at_1"] - baseline["chunk_recall_at_1"]
    recall_3_delta = rerank["chunk_recall_at_3"] - baseline["chunk_recall_at_3"]

    print("\n" + "=" * 80)
    print("Rerank 前后对比")
    print("指标                 no-rerank        rerank        delta")
    print(
        f"Chunk Recall@1     "
        f"{baseline['chunk_recall_at_1']:.2%}        "
        f"{rerank['chunk_recall_at_1']:.2%}      "
        f"{recall_1_delta:+.2%}"
    )
    print(
        f"Chunk Recall@3     "
        f"{baseline['chunk_recall_at_3']:.2%}        "
        f"{rerank['chunk_recall_at_3']:.2%}      "
        f"{recall_3_delta:+.2%}"
    )


def print_compare_query_rewrite_summary(
    baseline: dict,
    rewritten: dict,
) -> None:
    """打印 no-query-rewrite 与 query-rewrite 的指标对比。"""
    recall_1_delta = rewritten["chunk_recall_at_1"] - baseline["chunk_recall_at_1"]
    recall_3_delta = rewritten["chunk_recall_at_3"] - baseline["chunk_recall_at_3"]
    mrr_delta = rewritten["mrr_at_3"] - baseline["mrr_at_3"]

    print("\n" + "=" * 80)
    print("Query Rewrite 前后对比")
    print("指标                 no-rewrite       rewrite       delta")
    print(
        f"Chunk Recall@1     "
        f"{baseline['chunk_recall_at_1']:.2%}        "
        f"{rewritten['chunk_recall_at_1']:.2%}      "
        f"{recall_1_delta:+.2%}"
    )
    print(
        f"Chunk Recall@3     "
        f"{baseline['chunk_recall_at_3']:.2%}        "
        f"{rewritten['chunk_recall_at_3']:.2%}      "
        f"{recall_3_delta:+.2%}"
    )
    print(
        f"MRR@3              "
        f"{baseline['mrr_at_3']:.2%}        "
        f"{rewritten['mrr_at_3']:.2%}      "
        f"{mrr_delta:+.2%}"
    )

def print_compare_prompts_summary(v1_metrics: dict, v2_metrics: dict) -> None:
    """打印 Prompt V1 与 Prompt V2 的指标对比。"""
    tool_delta = v2_metrics["tool_call_pass_rate"] - v1_metrics["tool_call_pass_rate"]
    source_delta = v2_metrics["source_hit_rate"] - v1_metrics["source_hit_rate"]
    recall_1_delta = v2_metrics["chunk_recall_at_1"] - v1_metrics["chunk_recall_at_1"]
    recall_3_delta = v2_metrics["chunk_recall_at_3"] - v1_metrics["chunk_recall_at_3"]
    mrr_delta = v2_metrics["mrr_at_3"] - v1_metrics["mrr_at_3"]
    answer_delta = v2_metrics["answer_point_hit_rate"] - v1_metrics["answer_point_hit_rate"]
    faithfulness_delta = (
        v2_metrics["citation_faithfulness_rate"]
        - v1_metrics["citation_faithfulness_rate"]
    )

    print("\n" + "=" * 80)
    print("Prompt 版本对比：v1 vs v2")
    print("指标                    v1            v2            delta")
    print(
        f"Tool Call Pass Rate   "
        f"{v1_metrics['tool_call_pass_rate']:.2%}       "
        f"{v2_metrics['tool_call_pass_rate']:.2%}       "
        f"{tool_delta:+.2%}"
    )
    print(
        f"Source Hit Rate       "
        f"{v1_metrics['source_hit_rate']:.2%}       "
        f"{v2_metrics['source_hit_rate']:.2%}       "
        f"{source_delta:+.2%}"
    )
    print(
        f"Chunk Recall@1        "
        f"{v1_metrics['chunk_recall_at_1']:.2%}        "
        f"{v2_metrics['chunk_recall_at_1']:.2%}        "
        f"{recall_1_delta:+.2%}"
    )
    print(
        f"Chunk Recall@3        "
        f"{v1_metrics['chunk_recall_at_3']:.2%}        "
        f"{v2_metrics['chunk_recall_at_3']:.2%}        "
        f"{recall_3_delta:+.2%}"
    )
    print(
        f"MRR@3                 "
        f"{v1_metrics['mrr_at_3']:.2%}       "
        f"{v2_metrics['mrr_at_3']:.2%}       "
        f"{mrr_delta:+.2%}"
    )
    print(
        f"Answer Point Hit Rate "
        f"{v1_metrics['answer_point_hit_rate']:.2%}       "
        f"{v2_metrics['answer_point_hit_rate']:.2%}       "
        f"{answer_delta:+.2%}"
    )
    print(
        f"Citation Faithfulness "
        f"{v1_metrics['citation_faithfulness_rate']:.2%}       "
        f"{v2_metrics['citation_faithfulness_rate']:.2%}       "
        f"{faithfulness_delta:+.2%}"
    )

    print("\n结论建议：")
    if answer_delta > 0 or tool_delta > 0 or source_delta > 0:
        print("- Prompt V2 在当前评估集上带来正向收益，可以考虑作为默认 Prompt。")
    elif answer_delta == 0 and tool_delta == 0 and source_delta == 0:
        print("- Prompt V2 在关键指标上与 V1 持平，但约束更明确，适合作为更稳健的默认 Prompt。")
    else:
        print("- Prompt V2 在部分指标上下降，需要查看低质量样本后再决定是否设为默认。")

def print_compare_rerank_modes_summary(
    keyword_metrics: dict,
    cross_encoder_metrics: dict,
) -> None:
    """打印 keyword rerank 与 cross_encoder rerank 的逐题差异分析。

    这个函数解决的问题是：
    只看整体指标时，我们只能知道“涨了还是跌了”；
    但不知道具体是哪一道题导致变化。

    所以这里会把两次评估的 case_results 按 id 对齐，然后对比：
    - top3 chunk 是否变化
    - Recall@1 / Recall@3 / MRR@3 是否变化
    - 哪些题 cross_encoder 更好
    - 哪些题 keyword 更好
    - 哪些题只是排序变了，但指标没变
    """
    recall_1_delta = (
        cross_encoder_metrics["chunk_recall_at_1"]
        - keyword_metrics["chunk_recall_at_1"]
    )
    recall_3_delta = (
        cross_encoder_metrics["chunk_recall_at_3"]
        - keyword_metrics["chunk_recall_at_3"]
    )
    mrr_delta = cross_encoder_metrics["mrr_at_3"] - keyword_metrics["mrr_at_3"]
    answer_delta = (
        cross_encoder_metrics["answer_point_hit_rate"]
        - keyword_metrics["answer_point_hit_rate"]
    )
    faithfulness_delta = (
        cross_encoder_metrics["citation_faithfulness_rate"]
        - keyword_metrics["citation_faithfulness_rate"]
    )

    print("\n" + "=" * 80)
    print("Rerank Mode 对比：keyword vs cross_encoder")
    print("指标                    keyword       cross_encoder       delta")
    print(
        f"Chunk Recall@1        "
        f"{keyword_metrics['chunk_recall_at_1']:.2%}        "
        f"{cross_encoder_metrics['chunk_recall_at_1']:.2%}          "
        f"{recall_1_delta:+.2%}"
    )
    print(
        f"Chunk Recall@3        "
        f"{keyword_metrics['chunk_recall_at_3']:.2%}        "
        f"{cross_encoder_metrics['chunk_recall_at_3']:.2%}          "
        f"{recall_3_delta:+.2%}"
    )
    print(
        f"MRR@3                 "
        f"{keyword_metrics['mrr_at_3']:.2%}       "
        f"{cross_encoder_metrics['mrr_at_3']:.2%}         "
        f"{mrr_delta:+.2%}"
    )
    print(
        f"Answer Point Hit Rate "
        f"{keyword_metrics['answer_point_hit_rate']:.2%}       "
        f"{cross_encoder_metrics['answer_point_hit_rate']:.2%}         "
        f"{answer_delta:+.2%}"
    )
    print(
        f"Citation Faithfulness "
        f"{keyword_metrics['citation_faithfulness_rate']:.2%}       "
        f"{cross_encoder_metrics['citation_faithfulness_rate']:.2%}         "
        f"{faithfulness_delta:+.2%}"
    )

    keyword_cases = {
        case["id"]: case
        for case in keyword_metrics.get("case_results", [])
    }
    cross_cases = {
        case["id"]: case
        for case in cross_encoder_metrics.get("case_results", [])
    }

    improved_cases = []
    degraded_cases = []
    reordered_only_cases = []

    for case_id, keyword_case in keyword_cases.items():
        cross_case = cross_cases.get(case_id)
        if cross_case is None:
            continue

        recall_3_case_delta = (
            cross_case["recall_at_3"] - keyword_case["recall_at_3"]
        )
        recall_1_case_delta = (
            cross_case["recall_at_1"] - keyword_case["recall_at_1"]
        )
        mrr_case_delta = cross_case["mrr_at_3"] - keyword_case["mrr_at_3"]

        diff = {
            "id": case_id,
            "question": keyword_case["question"],
            "expected_chunk_ids": keyword_case["expected_chunk_ids"],
            "keyword_top3": keyword_case["actual_top3_chunk_ids"],
            "cross_encoder_top3": cross_case["actual_top3_chunk_ids"],
            "keyword_recall_at_1": keyword_case["recall_at_1"],
            "cross_encoder_recall_at_1": cross_case["recall_at_1"],
            "keyword_recall_at_3": keyword_case["recall_at_3"],
            "cross_encoder_recall_at_3": cross_case["recall_at_3"],
            "keyword_mrr_at_3": keyword_case["mrr_at_3"],
            "cross_encoder_mrr_at_3": cross_case["mrr_at_3"],
            "recall_1_delta": recall_1_case_delta,
            "recall_3_delta": recall_3_case_delta,
            "mrr_delta": mrr_case_delta,
        }

        # 这里主要用 Recall@3 判断“覆盖率”提升或下降；
        # 如果 Recall@3 相同，再看 Recall@1 和 MRR@3。
        if (
            recall_3_case_delta > 0
            or recall_1_case_delta > 0
            or mrr_case_delta > 0
        ):
            improved_cases.append(diff)
        elif (
            recall_3_case_delta < 0
            or recall_1_case_delta < 0
            or mrr_case_delta < 0
        ):
            degraded_cases.append(diff)
        elif keyword_case["actual_top3_chunk_ids"] != cross_case["actual_top3_chunk_ids"]:
            reordered_only_cases.append(diff)

    def print_case_group(title: str, cases: list[dict]) -> None:
        """打印一组逐题差异。"""
        print(f"\n{title}：")
        if not cases:
            print("无")
            return

        for case in cases:
            print("---------------------------------")
            print("id:", case["id"])
            print("question:", case["question"])
            print("expected_chunk_ids:", case["expected_chunk_ids"])
            print("keyword_top3:", case["keyword_top3"])
            print("cross_encoder_top3:", case["cross_encoder_top3"])
            print(
                "Recall@1:",
                case["keyword_recall_at_1"],
                "->",
                case["cross_encoder_recall_at_1"],
                f"({case['recall_1_delta']:+.4f})",
            )
            print(
                "Recall@3:",
                case["keyword_recall_at_3"],
                "->",
                case["cross_encoder_recall_at_3"],
                f"({case['recall_3_delta']:+.4f})",
            )
            print(
                "MRR@3:",
                case["keyword_mrr_at_3"],
                "->",
                case["cross_encoder_mrr_at_3"],
                f"({case['mrr_delta']:+.4f})",
            )

    print_case_group("cross_encoder 更好的样本", improved_cases)
    print_case_group("cross_encoder 变差的样本", degraded_cases)
    print_case_group("指标不变但 top3 顺序/内容变化的样本", reordered_only_cases)

    print("\n结论建议：")
    if recall_3_delta < 0 and mrr_delta >= 0:
        print(
            "- cross_encoder 的 MRR@3 没下降，说明核心正确 chunk 仍然靠前；"
            "但 Recall@3 下降，说明多 chunk 覆盖率变弱。"
        )
        print(
            "- 当前阶段建议保留 keyword 作为默认 rerank，"
            "cross_encoder 作为可选实验模式。"
        )
    elif recall_3_delta > 0 or mrr_delta > 0:
        print(
            "- cross_encoder 在当前评估集上带来正向收益，"
            "可以考虑进一步调 candidate_k 并作为默认候选。"
        )
    else:
        print(
            "- 两种 rerank 在当前评估集上整体接近。"
            "如果线上更关注速度和稳定性，默认 keyword 更合适；"
            "如果文档规模扩大、问题更口语化，可以继续观察 cross_encoder。"
        )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Agent / RAG eval")
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="关闭 rerank baseline，只评估 Chroma 原始向量检索顺序。",
    )
    parser.add_argument(
        "--compare-rerank",
        action="store_true",
        help="先关闭 rerank 跑一遍，再开启 rerank 跑一遍，并输出对比。",
    )
    parser.add_argument(
        "--no-query-rewrite",
        action="store_true",
        help="关闭 query rewrite，只评估原始 query 检索效果。",
    )
    parser.add_argument(
        "--compare-query-rewrite",
        action="store_true",
        help="先关闭 query rewrite 跑一遍，再开启 query rewrite 跑一遍，并输出对比。",
    )
    parser.add_argument(
        "--compare-rerank-modes",
        action="store_true",
        help="分别运行 keyword rerank 和 cross_encoder rerank，并输出逐题差异分析。",
    )
    parser.add_argument(
        "--rerank-mode",
        choices=["keyword", "cross_encoder"],
        default="keyword",
        help="选择 rerank 模式：keyword 表示词项重叠 baseline；cross_encoder 表示正式 CrossEncoder reranker。",
    )
    parser.add_argument(
        "--prompt-version",
        choices=["v1", "v2"],
        default="v1",
        help="选择 system prompt 版本：v1 为原始 prompt；v2 为优化后的 RAG grounding / 输出格式约束 prompt。",
    )
    parser.add_argument(
        "--compare-prompts",
        action="store_true",
        help="分别运行 Prompt V1 和 Prompt V2，并输出指标对比。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    questions = load_questions()

    compare_flags = [
        args.compare_rerank,
        args.compare_query_rewrite,
        args.compare_rerank_modes,
        args.compare_prompts,
    ]
    if sum(bool(flag) for flag in compare_flags) > 1:
        raise ValueError(
            "compare-rerank、compare-query-rewrite、compare-rerank-modes "
            "不能同时开启。"
        )

    default_use_query_rewrite = False
    default_use_rerank = not args.no_rerank
    default_rerank_mode = args.rerank_mode
    default_prompt_version = args.prompt_version

    if args.compare_rerank:
        baseline_metrics = evaluate_questions(
            questions,
            use_rerank=False,
            rerank_mode=default_rerank_mode,
            use_query_rewrite=default_use_query_rewrite,
            prompt_version=default_prompt_version,
            label="baseline_no_rerank",
        )
        rerank_metrics = evaluate_questions(
            questions,
            use_rerank=True,
            rerank_mode=default_rerank_mode,
            use_query_rewrite=default_use_query_rewrite,
            prompt_version=default_prompt_version,
            label=f"{default_rerank_mode}_rerank_baseline",
        )
        print_compare_summary(baseline_metrics, rerank_metrics)
    elif args.compare_query_rewrite:
        baseline_metrics = evaluate_questions(
            questions,
            use_rerank=default_use_rerank,
            rerank_mode=default_rerank_mode,
            use_query_rewrite=False,
            prompt_version=default_prompt_version,
            label="baseline_no_query_rewrite",
        )
        rewrite_metrics = evaluate_questions(
            questions,
            use_rerank=default_use_rerank,
            rerank_mode=default_rerank_mode,
            use_query_rewrite=True,
            prompt_version=default_prompt_version,
            label="query_rewrite_baseline",
        )
        print_compare_query_rewrite_summary(baseline_metrics, rewrite_metrics)
    elif args.compare_rerank_modes:
        keyword_metrics = evaluate_questions(
            questions,
            use_rerank=True,
            rerank_mode="keyword",
            use_query_rewrite=default_use_query_rewrite,
            prompt_version=default_prompt_version,
            label="keyword_rerank_baseline_no_rewrite",
            verbose=False,
        )
        cross_encoder_metrics = evaluate_questions(
            questions,
            use_rerank=True,
            rerank_mode="cross_encoder",
            use_query_rewrite=default_use_query_rewrite,
            prompt_version=default_prompt_version,
            label="cross_encoder_rerank_baseline_no_rewrite",
            verbose=False,
        )
        print_compare_rerank_modes_summary(
            keyword_metrics,
            cross_encoder_metrics,
        )
    elif args.compare_prompts:
        v1_metrics = evaluate_questions(
            questions,
            use_rerank=default_use_rerank,
            rerank_mode=default_rerank_mode,
            use_query_rewrite=default_use_query_rewrite,
            prompt_version="v1",
            label="prompt_v1_baseline",
            verbose=False,
        )
        v2_metrics = evaluate_questions(
            questions,
            use_rerank=default_use_rerank,
            rerank_mode=default_rerank_mode,
            use_query_rewrite=default_use_query_rewrite,
            prompt_version="v2",
            label="prompt_v2_optimized",
            verbose=False,
        )
        print_compare_prompts_summary(v1_metrics, v2_metrics)
    else:
        evaluate_questions(
            questions,
            use_rerank=default_use_rerank,
            rerank_mode=default_rerank_mode,
            prompt_version=default_prompt_version,
            use_query_rewrite=default_use_query_rewrite,
            label=(
                f"{default_rerank_mode}_rerank_baseline"
                if default_use_rerank
                else "baseline_no_rerank"
            )
            + f"_no_rewrite_{default_prompt_version}",
        )
