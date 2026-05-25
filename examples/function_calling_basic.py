# Function Calling 基础示例
import os
from dotenv import load_dotenv
import json
from openai import OpenAI

# 加载当前目录下的 .env。
# override=True 很重要：如果你的 shell/conda 环境里已经有 OPENAI_API_KEY，
# 默认 load_dotenv() 不会覆盖它，程序就会继续使用旧 key。
load_dotenv(dotenv_path=".env", override=True)

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    # 如果你在国内，使用的是第三方的API，可能需要配置 base_url，通常是中转服务的地址
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
)

# 计算器函数定义
def calculator(operation: str, a: float, b: float) -> dict:
    """执行基础数学运算"""
    operations = {
        'add': lambda x, y: x + y,
        'subtract': lambda x, y: x - y,
        'multiply': lambda x, y: x * y,
        'divide': lambda x, y: x / y if y != 0 else 'Error: Division by zero'
    }
    if operation in operations:
        try:
            result = operations[operation](a, b)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}
    else:
        return {"error": "Unknown operation"}    

# get_weather函数定义
def get_weather(city: str) -> dict:
    """查询指定城市的天气"""
    return {
        "city": city,
        "weather":"晴天",
        "temperature":"26°C"
    }
    


# 函数 Schema 定义
# 这个calculator_schema是给模型看的
# Function Calling的重点！！！
calculator_schema = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "执行基础数学运算",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "要执行的数学运算类型"
                },
                "a": {
                    "type": "number",
                    "description": "第一个操作数"
                },
                "b": {
                    "type": "number",
                    "description": "第二个操作数"
                }
            },
            "required": ["operation", "a", "b"]
        }
    }
}

weather_schema = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市的天气",
        "parameters": {
            "type": "object",
            "properties":{
                "city":{
                    "type": "string",
                    "description": "要查询天气的城市名称，例如：上海、北京、广州等"
                }
            },
            "required": ["city"]
        }
    }
}

# 可用函数映射
# 模型在上面的 schema看到可以调用 calculator，代码在这里映射然后python执行
# 这个 available_functions 是给程序看的，告诉代码如果模型调用了 calculator，就执行这个函数
available_functions = {
    "calculator": calculator,
    "get_weather": get_weather
}

# 这个函数是整个 demo 的主流程
def handle_function_calling(user_message: str) -> str:
    """处理 Function Calling 的完整流程"""
    try:
        # 第一次 API 调用
        response = client.chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            messages=[
                {"role": "system", "content": "你是一个有用的助手，可以执行数学计算和查询天气。"},
                {"role": "user", "content": user_message}
            ],
            tools=[calculator_schema, weather_schema],
            tool_choice="auto"
        )

        message = response.choices[0].message

        #这个 message.too_calls 可能为 None，所以需要判断模型是否调用了函数

        # 检查是否需要函数调用
        if message.tool_calls:
            # message.tool_calls 是一个列表，包含了模型请求调用的所有工具信息
            print(f"模型请求调用了{len(message.tool_calls)}个工具：")

            # 如果模型请求调用工具，就进入工具调用流程
            messages = [
                {"role": "system", "content": "你是一个有用的助手，可以执行数学计算和查询天气。"},
                {"role": "user", "content": user_message},
                message
            ]

            # 处理每个函数调用
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print("准备调用工具：", function_name)
                print("工具参数：", function_args)

                # 执行函数
                function_response = available_functions[function_name](**function_args)

                print("工具执行结果：", function_response)
                
                # 添加函数执行结果
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": json.dumps(function_response, ensure_ascii=False)
                })

            # 获取最终响应
            final_response = client.chat.completions.create(
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
                messages=messages,
                tools=[calculator_schema, weather_schema]
            )

            return final_response.choices[0].message.content
        else:
            return message.content

    except Exception as e:
        print(f"Function calling error: {e}")
        return "抱歉，处理您的请求时发生了错误。"

# 使用示例
if __name__ == "__main__":
    result = handle_function_calling("请帮我计算 15 加 27 的结果，然后告诉我上海今天天气怎么样？")
    print(result)
