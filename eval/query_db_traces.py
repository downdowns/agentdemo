"""查询 MySQL 中的 Agent Trace 记录。"""

import sys
from pathlib import Path


# 当前文件在 eval/ 目录下。
# database.py 在项目根目录。
# 所以这里把项目根目录加入 Python 模块搜索路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from database import get_connection


def query_recent_traces(limit: int = 10) -> list[dict]:
    """查询最近的 Agent Trace。"""

    sql = """
    SELECT
        trace_id,
        session_id,
        agent_type,
        user_query,
        duration_ms,
        success,
        error,
        created_at
    FROM agent_traces
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
                    "trace_id": row[0],
                    "session_id": row[1],
                    "agent_type": row[2],
                    "user_query": row[3],
                    "duration_ms": row[4],
                    "success": row[5],
                    "error": row[6],
                    "created_at": row[7],
                }
            )

        return results

    finally:
        connection.close()

def query_slow_traces(limit: int = 5) -> list[dict]:
    """查询耗时最长的 Agent Trace。"""

    sql = """
    SELECT
        trace_id,
        session_id,
        agent_type,
        user_query,
        duration_ms,
        success,
        error,
        created_at
    FROM agent_traces
    WHERE duration_ms IS NOT NULL
    ORDER BY duration_ms DESC
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
                    "trace_id": row[0],
                    "session_id": row[1],
                    "agent_type": row[2],
                    "user_query": row[3],
                    "duration_ms": row[4],
                    "success": row[5],
                    "error": row[6],
                    "created_at": row[7],
                }
            )

        return results

    finally:
        connection.close()

def query_failed_traces(limit: int = 10) -> list[dict]:
    """查询失败的 Agent Trace。"""

    sql = """
    SELECT
        trace_id,
        session_id,
        agent_type,
        user_query,
        duration_ms,
        success,
        error,
        created_at
    FROM agent_traces
    WHERE success = FALSE
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
                    "trace_id": row[0],
                    "session_id": row[1],
                    "agent_type": row[2],
                    "user_query": row[3],
                    "duration_ms": row[4],
                    "success": row[5],
                    "error": row[6],
                    "created_at": row[7],
                }
            )

        return results

    finally:
        connection.close()

def query_tool_calls_by_trace(trace_id: str) -> list[dict]:
    """根据 trace_id 查询工具调用记录。"""

    sql = """
    SELECT
        trace_id,
        session_id,
        tool_name,
        tool_args,
        tool_result,
        duration_ms,
        success,
        error,
        created_at
    FROM tool_call_logs
    WHERE trace_id = %s
    ORDER BY created_at ASC;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, (trace_id,))
            rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append(
                {
                    "trace_id": row[0],
                    "session_id": row[1],
                    "tool_name": row[2],
                    "tool_args": row[3],
                    "tool_result": row[4],
                    "duration_ms": row[5],
                    "success": row[6],
                    "error": row[7],
                    "created_at": row[8],
                }
            )

        return results

    finally:
        connection.close()

def query_messages_by_session(session_id: str, limit: int = 50) -> list[dict]:
    """根据 session_id 查询聊天历史。"""

    sql = """
    SELECT
        session_id,
        trace_id,
        role,
        content,
        created_at
    FROM chat_messages
    WHERE session_id = %s
    ORDER BY created_at ASC
    LIMIT %s;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, (session_id, limit))
            rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append(
                {
                    "session_id": row[0],
                    "trace_id": row[1],
                    "role": row[2],
                    "content": row[3],
                    "created_at": row[4],
                }
            )

        return results

    finally:
        connection.close()


if __name__ == "__main__":
    traces = query_recent_traces(limit=10)

    print("最近 10 条 Agent Trace：")
    print("-" * 80)

    for trace in traces:
        print("trace_id:", trace["trace_id"])
        print("session_id:", trace["session_id"])
        print("agent_type:", trace["agent_type"])
        print("user_query:", trace["user_query"])
        print("duration_ms:", trace["duration_ms"])
        print("success:", trace["success"])
        print("error:", trace["error"])
        print("created_at:", trace["created_at"])
        print("-" * 80)
    
    slow_traces = query_slow_traces(limit=5)

    print("\n慢请求 Top5：")
    print("-" * 80)

    for trace in slow_traces:
        print("trace_id:", trace["trace_id"])
        print("session_id:", trace["session_id"])
        print("user_query:", trace["user_query"])
        print("duration_ms:", trace["duration_ms"])
        print("success:", trace["success"])
        print("error:", trace["error"])
        print("created_at:", trace["created_at"])
        print("-" * 80)

    failed_traces = query_failed_traces(limit=10)

    print("\n失败请求：")
    print("-" * 80)

    if not failed_traces:
        print("暂无失败请求")
    else:
        for trace in failed_traces:
            print("trace_id:", trace["trace_id"])
            print("session_id:", trace["session_id"])
            print("user_query:", trace["user_query"])
            print("duration_ms:", trace["duration_ms"])
            print("success:", trace["success"])
            print("error:", trace["error"])
            print("created_at:", trace["created_at"])
            print("-" * 80)

    if traces:
        first_trace_id = traces[0]["trace_id"]
        tool_calls = query_tool_calls_by_trace(first_trace_id)

        print(f"\ntrace_id={first_trace_id} 的工具调用记录：")
        print("-" * 80)

        if not tool_calls:
            print("该 trace 暂无工具调用记录")
        else:
            for tool_call in tool_calls:
                print("tool_name:", tool_call["tool_name"])
                print("duration_ms:", tool_call["duration_ms"])
                print("success:", tool_call["success"])
                print("error:", tool_call["error"])
                print("created_at:", tool_call["created_at"])
                print("-" * 80)

    if traces:
        first_session_id = traces[0]["session_id"]
        messages = query_messages_by_session(first_session_id)

        print(f"\nsession_id={first_session_id} 的聊天历史：")
        print("-" * 80)

        if not messages:
            print("该 session 暂无聊天消息")
        else:
            for message in messages:
                print("role:", message["role"])
                print("content:", message["content"])
                print("trace_id:", message["trace_id"])
                print("created_at:", message["created_at"])
                print("-" * 80)