"""分析 Agent 日志，统计基础 Metrics。

这个脚本读取 logs/agent.log 中的 JSONL 日志，
将每一条 Agent trace 汇总成整体统计指标。

Trace 关注“一次请求发生了什么”；
Metrics 关注“很多次请求整体表现怎么样”。
"""

import json
from collections import Counter
from pathlib import Path

# 表示我们要分析的日志文件路径。
LOG_PATH = Path("logs/agent.log")


def load_logs(log_path: Path) -> list[dict]:
    """读取 JSONL 日志文件。

    JSONL 的特点：
    - 一行是一条 JSON
    - 适合持续追加写日志
    - 也方便逐行读取分析
    """
    records = []

    if not log_path.exists():
        return records

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records

def analyze_basic_metrics(records: list[dict]) -> dict:
    """统计 Agent 日志的基础指标。

    注意：
    早期日志可能还没有 success / duration_ms 字段，
    所以这里不能直接用 record["success"]，
    而是用 record.get(...) 做兼容处理。
    """
    total_count = len(records)

    # 成功请求：明确 success=True 的日志。
    success_records = [
        record for record in records
        if record.get("success") is True
    ]

    # 失败请求：明确 success=False 的日志。
    failed_records = [
        record for record in records
        if record.get("success") is False
    ]

    # 未知状态请求：老日志可能没有 success 字段。
    unknown_records = [
        record for record in records
        if "success" not in record
    ]

    known_status_count = len(success_records) + len(failed_records)

    # 成功率只在“有 success 字段的日志”里计算。
    # 否则老日志会干扰统计。
    if known_status_count == 0:
        success_rate = 0
    else:
        success_rate = len(success_records) / known_status_count

    # 只统计有 duration_ms 的日志。
    duration_values = [
        record.get("duration_ms")
        for record in records
        if isinstance(record.get("duration_ms"), (int, float))
    ]

    if len(duration_values) == 0:
        avg_duration_ms = 0
    else:
        avg_duration_ms = sum(duration_values) / len(duration_values)
    
    # 统计每条日志中的模型调用次数。
    # 新日志中 model_calls 是一个 list，每个元素代表一轮模型调用。
    model_call_counts = [
        len(record.get("model_calls", []))
        for record in records
        if "model_calls" in record and isinstance(record.get("model_calls", []), list)
    ]

    if len(model_call_counts) == 0:
        avg_model_call_count = 0
    else:
        avg_model_call_count = sum(model_call_counts) / len(model_call_counts)

    # 统计每条日志中的工具调用次数。
    # tool_calls 是一个 list，每个元素代表一次工具调用。
    tool_call_counts = [
        len(record["tool_calls"])
        for record in records
        if "tool_calls" in record and isinstance(record["tool_calls"], list)
    ]

    if len(tool_call_counts) == 0:
        avg_tool_call_count = 0
    else:
        avg_tool_call_count = sum(tool_call_counts) / len(tool_call_counts)

    # 统计所有工具名出现次数，用于生成工具调用排行榜。
    tool_name_counter = Counter()

    # 统计不同 Agent 类型的请求数量。
    # 新日志中 agent_type 可能是 manual / langgraph；
    # 老日志没有 agent_type，统一记为 unknown。
    agent_type_counter = Counter()

    for record in records:
        agent_type = record.get("agent_type", "unknown")
        agent_type_counter[agent_type] += 1

        tool_calls = record.get("tool_calls", [])

        if not isinstance(tool_calls, list):
            continue

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")

            if tool_name:
                tool_name_counter[tool_name] += 1

    return {
        "total_count": total_count,
        "known_status_count": known_status_count,
        "success_count": len(success_records),
        "failed_count": len(failed_records),
        "unknown_count": len(unknown_records),
        "success_rate": success_rate,
        "duration_count": len(duration_values),
        "avg_duration_ms": avg_duration_ms,
        "avg_model_call_count": avg_model_call_count,
        "avg_tool_call_count": avg_tool_call_count,
        "model_call_count_sample_size": len(model_call_counts),
        "tool_call_count_sample_size": len(tool_call_counts),
        "total_tool_call_count": sum(tool_name_counter.values()),
        "tool_name_counter": tool_name_counter,
        "agent_type_counter": agent_type_counter,
    }

def analyze_slow_and_failed(records: list[dict]) -> dict:
    """统计慢请求和失败请求。"""
    # 只保留有 duration_ms 的记录，方便排序。
    duration_records = [
        record for record in records
        if isinstance(record.get("duration_ms"), (int, float))
    ]

    # 按耗时从大到小排序，取前 5 条。
    slow_top5 = sorted(
        duration_records,
        key=lambda x: x.get("duration_ms", 0),
        reverse=True,
    )[:5]

    # 失败请求：success=False 的记录。
    failed_records = [
        record for record in records
        if record.get("success") is False
    ]

    failed_trace_ids = [
        record.get("trace_id")
        for record in failed_records
        if record.get("trace_id")
    ]

    return {
        "slow_top5": slow_top5,
        "failed_trace_ids": failed_trace_ids,
    }

def analyze_agent_type_details(records: list[dict]) -> dict[str, dict]:
    """按 agent_type 统计更细的运行指标。"""
    grouped_records: dict[str, list[dict]] = {}

    for record in records:
        agent_type = record.get("agent_type", "unknown")
        grouped_records.setdefault(agent_type, []).append(record)

    agent_type_details: dict[str, dict] = {}

    for agent_type, subset in grouped_records.items():
        total_count = len(subset)

        success_records = [
            record for record in subset
            if record.get("success") is True
        ]
        failed_records = [
            record for record in subset
            if record.get("success") is False
        ]

        known_status_count = len(success_records) + len(failed_records)
        success_rate = (
            len(success_records) / known_status_count
            if known_status_count
            else 0
        )

        duration_values = [
            record.get("duration_ms")
            for record in subset
            if isinstance(record.get("duration_ms"), (int, float))
        ]
        avg_duration_ms = (
            sum(duration_values) / len(duration_values)
            if duration_values
            else 0
        )

        model_call_counts = [
            len(record["model_calls"])
            for record in subset
            if "model_calls" in record and isinstance(record.get("model_calls"), list)
        ]
        avg_model_call_count = (
            sum(model_call_counts) / len(model_call_counts)
            if model_call_counts
            else 0
        )

        tool_call_counts = [
            len(record["tool_calls"])
            for record in subset
            if "tool_calls" in record and isinstance(record.get("tool_calls"), list)
        ]
        avg_tool_call_count = (
            sum(tool_call_counts) / len(tool_call_counts)
            if tool_call_counts
            else 0
        )

        agent_type_details[agent_type] = {
            "total_count": total_count,
            "success_count": len(success_records),
            "failed_count": len(failed_records),
            "success_rate": success_rate,
            "avg_duration_ms": avg_duration_ms,
            "avg_model_call_count": avg_model_call_count,
            "avg_tool_call_count": avg_tool_call_count,
            "model_call_count_sample_size": len(model_call_counts),
            "tool_call_count_sample_size": len(tool_call_counts),
        }

    return agent_type_details

def print_report(metrics: dict, extra_metrics: dict, agent_type_details: dict[str, dict]) -> None:
    """打印面试可讲的 Metrics 报告。"""
    print("\n========== Agent 运行报告 ==========")
    print(f"日志总条数：{metrics['total_count']}")
    print(f"成功请求数：{metrics['success_count']}")
    print(f"失败请求数：{metrics['failed_count']}")
    print(f"成功率：{metrics['success_rate']:.2%}")
    print(f"平均总耗时：{metrics['avg_duration_ms']:.2f} ms")
    print(
        f"平均模型调用次数：{metrics['avg_model_call_count']:.2f} "
        f"（统计样本数：{metrics['model_call_count_sample_size']}）"
    )
    print(
        f"平均工具调用次数：{metrics['avg_tool_call_count']:.2f} "
        f"（统计样本数：{metrics['tool_call_count_sample_size']}）"
    )

    print("\n========== Agent 类型分布 ==========")
    if metrics["agent_type_counter"]:
        for agent_type, count in metrics["agent_type_counter"].most_common():
            print(f"{agent_type}: {count}")
    else:
        print("暂无 agent_type 记录")

    print("\n========== 按 Agent 类型统计 ==========")
    if agent_type_details:
        preferred_order = ["manual", "langgraph", "unknown"]
        printed = set()

        def print_agent_summary(agent_type: str) -> None:
            summary = agent_type_details.get(agent_type)
            if not summary:
                return

            printed.add(agent_type)
            print(
                f"{agent_type}: "
                f"请求数={summary['total_count']}，"
                f"成功率={summary['success_rate']:.2%}，"
                f"平均总耗时={summary['avg_duration_ms']:.2f} ms，"
                f"平均模型调用次数={summary['avg_model_call_count']:.2f} "
                f"（样本数：{summary['model_call_count_sample_size']}），"
                f"平均工具调用次数={summary['avg_tool_call_count']:.2f} "
                f"（样本数：{summary['tool_call_count_sample_size']}）"
            )

        for agent_type in preferred_order:
            print_agent_summary(agent_type)

        for agent_type in sorted(agent_type_details):
            if agent_type not in printed:
                print_agent_summary(agent_type)
    else:
        print("暂无按 Agent 类型统计数据")

    print("\n========== 工具使用情况 ==========")
    if metrics["tool_name_counter"]:
        for tool_name, count in metrics["tool_name_counter"].most_common():
            print(f"{tool_name}: {count}")
    else:
        print("暂无工具调用记录")

    print("\n========== 慢请求 Top5 ==========")
    if extra_metrics["slow_top5"]:
        for idx, record in enumerate(extra_metrics["slow_top5"], start=1):
            print(
                f"{idx}. trace_id={record.get('trace_id')} "
                f"duration_ms={record.get('duration_ms')} "
                f"question={record.get('user_query')}"
            )
    else:
        print("暂无慢请求记录")

    print("\n========== 失败请求 ==========")
    if extra_metrics["failed_trace_ids"]:
        for trace_id in extra_metrics["failed_trace_ids"]:
            print(f"- {trace_id}")
    else:
        print("暂无失败请求")

if __name__ == "__main__":
    records = load_logs(LOG_PATH)
    metrics = analyze_basic_metrics(records)
    extra_metrics = analyze_slow_and_failed(records)
    agent_type_details = analyze_agent_type_details(records)

    print_report(metrics, extra_metrics, agent_type_details)
