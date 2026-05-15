"""工具 Schema：给模型看的“工具说明书”。"""


# search_docs_schema 是给模型看的，告诉模型：
# 有一个 search_docs 工具，可以用来搜索本地知识库，需要 query 参数。
search_docs_schema = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "搜索本地知识库，适合回答和 RAG、Agent、Function Calling、个人资料相关的问题。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用户的问题或要搜索的关键词。",
                },
                "k": {
                    "type": "integer",
                    "description": "要返回的相关文档片段数量，通常取 2 或 3。",
                },
            },
            "required": ["query"],
        },
    },
}

# calculator_schema 是给模型看的，告诉模型：
# 有一个 calculator 工具，可以做四则运算，需要 operation、a、b 三个参数。
calculator_schema = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "执行基础数学运算，包括加法、减法、乘法和除法。",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "要执行的数学运算类型。",
                },
                "a": {
                    "type": "number",
                    "description": "第一个操作数。",
                },
                "b": {
                    "type": "number",
                    "description": "第二个操作数。",
                },
            },
            "required": ["operation", "a", "b"],
        },
    },
}

# get_weather_schema 是给模型看的，告诉模型：
# 有一个 get_weather 工具，可以查询城市天气，需要 city 参数。
get_weather_schema = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市的天气。当前是 demo 模拟数据，不是真实实时天气。",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "要查询天气的城市名称，例如：上海、北京、广州、深圳。",
                }
            },
            "required": ["city"],
        },
    },
}

# 传给模型看的工具列表。
TOOLS = [search_docs_schema, calculator_schema, get_weather_schema]
