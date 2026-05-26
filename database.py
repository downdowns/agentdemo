"""MySQL 持久化模块。

这个文件负责：
1. 读取 MySQL 配置；
2. 创建数据库连接；
3. 初始化表结构；
4. 保存 LangGraph Agent 的会话、消息、Trace、工具调用记录和用户反馈。

这些数据用于支撑 Agent 可观测性、问题排查和 feedback 数据飞轮。
"""

import os
import json
import uuid
import pymysql
from dataclasses import dataclass

from dotenv import load_dotenv

# 读取项目根目录的 .env文件
load_dotenv(dotenv_path=".env", override=True)


@dataclass
class MySQLConfig:
    """MySQL 连接配置。

    dataclass 可以理解为“专门用来装数据的小类”。
    """

    host: str
    port: int
    user: str
    password: str
    database: str
    charset: str


def get_mysql_config() -> MySQLConfig:
    """从环境变量读取 MySQL 配置。"""

    return MySQLConfig(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "rag_agent"),
        charset=os.getenv("MYSQL_CHARSET", "utf8mb4"),
    )

def get_connection():
    """创建 MySQL 数据库连接。"""

    config = get_mysql_config()

    return pymysql.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.database,
        charset=config.charset,
        # SQL 执行后，不自动提交，需要手动 connection.commit()
        autocommit=False,
    )

def init_db():
    """初始化数据库表。"""

    create_chat_sessions_sql = """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        session_id VARCHAR(128) NOT NULL UNIQUE,
        agent_type VARCHAR(32) DEFAULT 'langgraph',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    );
    """

    create_chat_messages_sql = """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        session_id VARCHAR(128) NOT NULL,
        trace_id VARCHAR(64),
        role VARCHAR(32) NOT NULL,
        content TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

        INDEX idx_session_id (session_id),
        INDEX idx_trace_id (trace_id),
        INDEX idx_created_at (created_at)
    );
    """

    create_agent_traces_sql = """
    CREATE TABLE IF NOT EXISTS agent_traces (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        trace_id VARCHAR(64) NOT NULL UNIQUE,
        session_id VARCHAR(128),
        agent_type VARCHAR(32) DEFAULT 'langgraph',
        user_query TEXT NOT NULL,
        answer TEXT,
        sources JSON,
        model_calls JSON,
        quality_check JSON,
        duration_ms INT,
        success BOOLEAN DEFAULT TRUE,
        error TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

        INDEX idx_session_id (session_id),
        INDEX idx_success (success),
        INDEX idx_duration_ms (duration_ms),
        INDEX idx_created_at (created_at)
    );
    """

    create_tool_call_logs_sql = """
    CREATE TABLE IF NOT EXISTS tool_call_logs (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        trace_id VARCHAR(64) NOT NULL,
        session_id VARCHAR(128),
        tool_name VARCHAR(64) NOT NULL,
        tool_args JSON,
        tool_result JSON,
        duration_ms INT,
        success BOOLEAN DEFAULT TRUE,
        error TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

        INDEX idx_trace_id (trace_id),
        INDEX idx_session_id (session_id),
        INDEX idx_tool_name (tool_name),
        INDEX idx_duration_ms (duration_ms),
        INDEX idx_success (success)
    );
    """

    create_feedback_logs_sql = """
    CREATE TABLE IF NOT EXISTS feedback_logs (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        feedback_id VARCHAR(64) NOT NULL UNIQUE,
        trace_id VARCHAR(64) NOT NULL,
        session_id VARCHAR(128),
        rating VARCHAR(16) NOT NULL,
        comment TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

        INDEX idx_trace_id (trace_id),
        INDEX idx_session_id (session_id),
        INDEX idx_rating (rating),
        INDEX idx_created_at (created_at)
    );
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(create_chat_sessions_sql)
            cursor.execute(create_chat_messages_sql)
            cursor.execute(create_agent_traces_sql)
            cursor.execute(create_tool_call_logs_sql)
            cursor.execute(create_feedback_logs_sql)

        connection.commit()
        print("All database tables initialized")

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

def save_chat_session(session_id: str, agent_type: str = "langgraph") -> None:
    """保存或更新会话记录。"""

    sql = """
    INSERT INTO chat_sessions (session_id, agent_type)
    VALUES (%s, %s)
    ON DUPLICATE KEY UPDATE
        agent_type = VALUES(agent_type),
        updated_at = CURRENT_TIMESTAMP;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, (session_id, agent_type))

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def _to_json(value):
    """把 Python 对象转换成 MySQL JSON 字段可以接收的字符串。

    MySQL 的 JSON 字段接收的是合法 JSON 字符串。
    例如 Python 里的 list/dict：
        [{"source": "xxx"}]
    需要转成：
        '[{"source": "xxx"}]'

    如果 value 是 None，就直接返回 None，表示数据库里存 NULL。
    """

    if value is None:
        return None

    return json.dumps(value, ensure_ascii=False)


def save_chat_message(
    session_id: str,
    role: str,
    content: str,
    trace_id: str | None = None,
) -> None:
    """保存一条聊天消息。

    chat_messages 表保存的是“消息级别”的数据。
    一次 Agent 请求通常至少会保存两条消息：
    - role=user：用户问题
    - role=assistant：模型最终回答
    """

    sql = """
    INSERT INTO chat_messages (session_id, trace_id, role, content)
    VALUES (%s, %s, %s, %s);
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, (session_id, trace_id, role, content))

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def save_agent_trace(
    trace_id: str,
    session_id: str,
    user_query: str,
    answer: str | None = None,
    sources: list[dict] | None = None,
    model_calls: list[dict] | None = None,
    quality_check: dict | None = None,
    duration_ms: int | None = None,
    success: bool = True,
    error: str | None = None,
    agent_type: str = "langgraph",
) -> None:
    """保存一次 LangGraph Agent 调用的整体 Trace。

    agent_traces 表保存的是“请求级别”的数据：
    - 这次请求的 trace_id 是什么
    - 属于哪个 session
    - 用户问了什么
    - 最终回答是什么
    - 检索来源 sources 是什么
    - 模型调用记录 model_calls 是什么
    - quality_check 结果是什么
    - 总耗时和成功失败状态

    trace_id 是唯一键，所以这里使用 ON DUPLICATE KEY UPDATE：
    如果同一个 trace_id 已经存在，就更新原记录，避免重复插入报错。
    """

    sql = """
    INSERT INTO agent_traces (
        trace_id,
        session_id,
        agent_type,
        user_query,
        answer,
        sources,
        model_calls,
        quality_check,
        duration_ms,
        success,
        error
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        session_id = VALUES(session_id),
        agent_type = VALUES(agent_type),
        user_query = VALUES(user_query),
        answer = VALUES(answer),
        sources = VALUES(sources),
        model_calls = VALUES(model_calls),
        quality_check = VALUES(quality_check),
        duration_ms = VALUES(duration_ms),
        success = VALUES(success),
        error = VALUES(error);
    """

    params = (
        trace_id,
        session_id,
        agent_type,
        user_query,
        answer,
        _to_json(sources),
        _to_json(model_calls),
        _to_json(quality_check),
        duration_ms,
        success,
        error,
    )

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def save_tool_call_log(
    trace_id: str,
    session_id: str,
    tool_name: str,
    tool_args: dict | None = None,
    tool_result=None,
    duration_ms: int | None = None,
    success: bool = True,
    error: str | None = None,
) -> None:
    """保存一次工具调用记录。

    tool_call_logs 表保存的是“工具调用级别”的数据。
    一次 Agent 请求可能调用多个工具，所以一个 trace_id 可以对应多条工具记录。
    """

    sql = """
    INSERT INTO tool_call_logs (
        trace_id,
        session_id,
        tool_name,
        tool_args,
        tool_result,
        duration_ms,
        success,
        error
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    """

    params = (
        trace_id,
        session_id,
        tool_name,
        _to_json(tool_args),
        _to_json(tool_result),
        duration_ms,
        success,
        error,
    )

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

def save_feedback(
    trace_id: str,
    rating: str,
    session_id: str | None = None,
    comment: str | None = None,
    feedback_id: str | None = None,
) -> str:
    """保存用户对某次 Agent 回答的反馈。

    feedback_logs 表保存的是“用户反馈级别”的数据：
    - trace_id：用户评价的是哪一次 Agent 请求
    - session_id：这次反馈属于哪个会话
    - rating：up / down
    - comment：用户补充说明

    返回 feedback_id，方便 API 返回给前端或调用方。
    """

    rating = rating.strip().lower()

    if rating not in {"up", "down"}:
        raise ValueError("rating 只能是 up 或 down")

    if feedback_id is None:
        feedback_id = uuid.uuid4().hex

    sql = """
    INSERT INTO feedback_logs (
        feedback_id,
        trace_id,
        session_id,
        rating,
        comment
    )
    VALUES (%s, %s, %s, %s, %s);
    """

    params = (
        feedback_id,
        trace_id,
        session_id,
        rating,
        comment,
    )

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)

        connection.commit()
        return feedback_id

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


if __name__ == "__main__":
    config = get_mysql_config()
    print(
        "MySQL config:",
        {
            "host": config.host,
            "port": config.port,
            "user": config.user,
            "database": config.database,
            "charset": config.charset,
        },
    )

    connection = get_connection()
    print("MySQL connection success")
    connection.close()

    init_db()
