# Python 环境下的 Function Calling 完整实现
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

tools = [calculator_schema, weather_schema]

# 这个函数是整个 demo 的主流程
def handle_function_calling(user_message: str) -> str:
    """处理 Function Calling 的Agent Loop流程"""
    try:
        # 执行完所有工具后，messages变成
        """
        system: 你是一个有用的助手
        user: 用户问题
        assistant: 我要调用 calculator 和 get_weather
        tool: calculator 的结果是 42
        tool: get_weather 的结果是 晴天 26°C
        """
        messages = [
            {"role": "system", "content": "你是一个有用的助手，可以执行数学计算和查询天气。"},
            {"role": "user", "content": user_message}
        ]

        round_num = 1

        print("\n========== Agent Loop 开始 ==========")
        print("用户输入：", user_message)

        while True:
            print(f"\n========== 第 {round_num} 轮：调用模型 ==========")
            print("当前 messages 条数：", len(messages))

            response = client.chat.completions.create(
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            # 取出模型回复，tool_calls在这里面
            message = response.choices[0].message

            if not message.tool_calls:
                print(f"\n========== 第 {round_num} 轮：模型没有继续调用工具 ==========")
                print("模型最终回答：", message.content)
                print("========== Agent Loop 结束 ==========\n")
                return message.content
            
            # 这一步非常关键，因为 message 里包含了模型刚才的tool_call请求，你需要把它加入对话历史，告诉下一轮模型：你刚刚请求调用了这些工具。后面再追加工具结果，模型才能正确对应起来
            """
            messages是Agent 的上下文记忆
            每一轮都要把这些内容保存进去：
            """
            messages.append(message)

            print(f"\n第 {round_num} 轮：模型请求调用了 {len(message.tool_calls)} 个工具")

            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print("\n--- 准备执行工具 ---")
                print("tool_call_id：", tool_call.id)
                print("工具名称：", function_name)
                print("工具参数：", function_args)

                function_response = available_functions[function_name](**function_args)

                print("工具执行结果：", function_response)

                # 把工具执行结果返回给模型
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": json.dumps(function_response, ensure_ascii=False)
                })

                print("工具结果已追加到 messages")

            print(f"\n第 {round_num} 轮结束，准备进入下一轮模型调用")
            round_num += 1

    except Exception as e:
        print(f"Function calling error: {e}")
        return "抱歉，处理您的请求时发生了错误。"

# 使用示例
if __name__ == "__main__":
    result = handle_function_calling("请帮我计算 15 加 27 的结果，然后告诉我上海今天天气怎么样？")
    print(result)
