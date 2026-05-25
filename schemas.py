"""工具 Schema：给模型看的“工具说明书”。

这个文件不执行真实工具，只负责告诉模型：
- 有哪些工具
- 每个工具适合做什么
- 每个工具需要哪些参数
- 参数类型和必填字段是什么

注意区分：
- schemas.py：给模型看的工具说明
- tools.py：给 Python 程序执行的真实函数

工具 schema 的描述质量会直接影响模型是否正确调用工具。
例如 description 写得不清楚，模型可能该调用时不调用，或者乱调用。
"""


# search_docs_schema 是给模型看的，告诉模型：
# 有一个 search_docs 工具，可以用来搜索本地知识库，需要 query 参数。
#
# description 很关键：
# 它决定模型在什么情况下会选择这个工具。
# 当前描述里写了适合 RAG / Agent / Function Calling / 个人资料相关问题，
# 是为了引导模型遇到知识库相关问题时调用 search_docs。
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
#
# enum 用来约束 operation 只能是四种值：
# add / subtract / multiply / divide
#
# 这能减少模型生成非法参数的概率。
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
#
# 当前天气工具是 demo 模拟数据，不是真实实时 API。
# 真实业务中可以把 tools.py 中的 get_weather 替换为真实 HTTP API 调用，
# schema 通常不需要大改。
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
#
# 在 agent.py / LangGraph Agent 中会传给模型：
# llm.invoke(..., tools=TOOLS, tool_choice="auto")
#
# tool_choice="auto" 表示让模型自己判断是否调用工具。
TOOLS = [search_docs_schema, calculator_schema, get_weather_schema]
