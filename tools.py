"""Agent 可以调用的真实 Python 工具函数。"""

from typing import Any

from vector_store import vector_store


def search_docs(query: str, k: int = 2) -> list[dict[str, str]]:
    """
    搜索本地知识库。

    参数：
    - query：用户问题或关键词
    - k：返回最相关的前 k 个文档片段

    返回：
    - source：文档来源文件名
    - content：检索到的文档内容
    """
    # similarity_search 会把 query 转成向量，
    # 再从 Chroma 里找出语义最相近的 k 个文档片段。
    retrieved_docs = vector_store.similarity_search(query, k=k)

    # 把 LangChain Document 对象转换成普通 dict，方便 json.dumps 和工具返回。
    results: list[dict[str, str]] = []
    for doc in retrieved_docs:
        results.append(
            {
                "source": doc.metadata.get("source", "unknown"),
                "content": doc.page_content,
            }
        )

    # 返回检索结果列表。
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
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero",
    }

    # 如果模型传入了不支持的 operation，就返回错误信息。
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
    return {
        "city": city,
        "weather": weather_info["weather"],
        "temperature": weather_info["temperature"],
        "note": "这是 demo 模拟天气，不是真实实时天气。",
    }


# 给 Python 程序看的工具映射。
# 模型只会返回工具名和参数；真正执行哪个 Python 函数，由这里决定。
AVAILABLE_FUNCTIONS = {
    "search_docs": search_docs,
    "calculator": calculator,
    "get_weather": get_weather,
}
