"""查询 MySQL 中的用户反馈数据。

这个脚本是 Feedback 数据飞轮的最小分析工具。

它解决的问题是：
1. 当前一共有多少条用户反馈？
2. 点赞 / 点踩分别有多少？
3. 好评率是多少？
4. 最近有哪些反馈？
5. 最近的差评样本对应的用户问题和 Agent 回答是什么？

运行：
    python eval/query_feedback.py
"""

import sys
from pathlib import Path
from typing import Any


# 当前文件在 eval/ 目录下，database.py 在项目根目录。
# 所以这里把项目根目录加入 Python 模块搜索路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from database import get_connection


def query_feedback_summary() -> dict[str, Any]:
    """统计反馈总数、点赞数、点踩数和好评率。"""

    sql = """
    SELECT
        COUNT(*) AS total_count,
        SUM(CASE WHEN rating = 'up' THEN 1 ELSE 0 END) AS up_count,
        SUM(CASE WHEN rating = 'down' THEN 1 ELSE 0 END) AS down_count
    FROM feedback_logs;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            row = cursor.fetchone()

        total_count = int(row[0] or 0)
        up_count = int(row[1] or 0)
        down_count = int(row[2] or 0)
        known_rating_count = up_count + down_count

        positive_rate = (
            up_count / known_rating_count
            if known_rating_count
            else 0
        )

        return {
            "total_count": total_count,
            "up_count": up_count,
            "down_count": down_count,
            "positive_rate": positive_rate,
        }

    finally:
        connection.close()


def query_recent_feedback(limit: int = 10) -> list[dict[str, Any]]:
    """查询最近的用户反馈。"""

    sql = """
    SELECT
        feedback_id,
        trace_id,
        session_id,
        rating,
        comment,
        created_at
    FROM feedback_logs
    ORDER BY created_at DESC
    LIMIT %s;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, (limit,))
            rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append(
                {
                    "feedback_id": row[0],
                    "trace_id": row[1],
                    "session_id": row[2],
                    "rating": row[3],
                    "comment": row[4],
                    "created_at": row[5],
                }
            )

        return results

    finally:
        connection.close()


def query_negative_feedback_with_trace(limit: int = 10) -> list[dict[str, Any]]:
    """查询最近差评，并关联 agent_traces 中的问题和回答。

    这一步是数据飞轮的关键：
    - feedback_logs 里知道用户点踩了哪次回答；
    - agent_traces 里保存了那次请求的问题、回答、耗时和状态；
    - 两张表通过 trace_id 关联。
    """

    sql = """
    SELECT
        f.feedback_id,
        f.trace_id,
        f.session_id,
        f.rating,
        f.comment,
        f.created_at,
        t.user_query,
        t.answer,
        t.duration_ms,
        t.success,
        t.error
    FROM feedback_logs f
    LEFT JOIN agent_traces t
        ON f.trace_id = t.trace_id
    WHERE f.rating = 'down'
    ORDER BY f.created_at DESC
    LIMIT %s;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, (limit,))
            rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append(
                {
                    "feedback_id": row[0],
                    "trace_id": row[1],
                    "session_id": row[2],
                    "rating": row[3],
                    "comment": row[4],
                    "created_at": row[5],
                    "user_query": row[6],
                    "answer": row[7],
                    "duration_ms": row[8],
                    "success": row[9],
                    "error": row[10],
                }
            )

        return results

    finally:
        connection.close()


def print_feedback_report() -> None:
    """打印反馈数据飞轮报告。"""

    summary = query_feedback_summary()

    print("\n========== Feedback 数据飞轮报告 ==========")
    print(f"反馈总数：{summary['total_count']}")
    print(f"点赞数 up：{summary['up_count']}")
    print(f"点踩数 down：{summary['down_count']}")
    print(f"好评率：{summary['positive_rate']:.2%}")

    recent_feedback = query_recent_feedback(limit=10)

    print("\n========== 最近 10 条反馈 ==========")
    if not recent_feedback:
        print("暂无反馈")
    else:
        for item in recent_feedback:
            print("-" * 80)
            print("feedback_id:", item["feedback_id"])
            print("trace_id:", item["trace_id"])
            print("session_id:", item["session_id"])
            print("rating:", item["rating"])
            print("comment:", item["comment"])
            print("created_at:", item["created_at"])

    negative_feedback = query_negative_feedback_with_trace(limit=10)

    print("\n========== 最近差评样本 Top10 ==========")
    if not negative_feedback:
        print("暂无差评样本")
    else:
        for item in negative_feedback:
            print("-" * 80)
            print("feedback_id:", item["feedback_id"])
            print("trace_id:", item["trace_id"])
            print("session_id:", item["session_id"])
            print("comment:", item["comment"])
            print("question:", item["user_query"])
            print("answer:", item["answer"])
            print("duration_ms:", item["duration_ms"])
            print("success:", item["success"])
            print("error:", item["error"])
            print("created_at:", item["created_at"])


if __name__ == "__main__":
    print_feedback_report()
