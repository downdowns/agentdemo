"""最小 Agent 评估脚本。

当前评估目标：
    检查 Agent 是否调用了期望的工具。

评估流程：
    1. 读取 eval/questions.json
    2. 对每个问题调用 run_agent()
    3. 从 Agent 返回的 tool_calls 中提取实际工具名
    4. 判断 expected_tools 是否都出现在 actual_tools 中
    5. 输出每道题结果和总体通过率

运行方式：
    python eval/run_eval.py
"""

import json
import sys
from pathlib import Path

# run_eval.py 在 eval 目录中，agent.py 在项目根目录。
# 为了支持 `python eval/run_eval.py` 这种运行方式，需要把项目根目录加入 sys.path。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from agent import run_agent

# 当前 run_eval.py 所在文件夹 / questions.json。
# 这样不管从哪个目录启动脚本，都能正确找到评估问题集。
QUESTIONS_PATH = Path(__file__).parent / "questions.json"


def load_questions() -> list[dict]:
    """读取评估问题集。

    questions.json 的每一条数据包含：
    - id：问题编号
    - question：用户问题
    - expected_tools：期望 Agent 调用的工具列表
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


def is_pass(expected_tools: list[str], actual_tools: list[str]) -> bool:
    """判断期望工具是否都被实际调用。

    当前采用“宽松判断”：
    只要 expected_tools 中的工具都出现在 actual_tools 中，就算通过。

    例：
    expected_tools = ["search_docs"]
    actual_tools = ["search_docs", "calculator"]
    当前会判定为通过。

    后续如果要更严格，可以增加“是否多调工具”的判断。
    """
    for tool in expected_tools:
        if tool not in actual_tools:
            return False

    return True


if __name__ == "__main__":
    questions = load_questions()

    # total：总题数；passed_count：通过题数。
    total = len(questions)
    passed_count = 0

    print(f"共加载 {total} 条评估问题")

    for item in questions:
        print("\n---------------------------------")
        print("id:", item["id"])
        print("question:", item["question"])
        print("expected_tools:", item["expected_tools"])

        result = run_agent(item["question"])

        # Agent 返回的是完整工具调用记录；
        # 评估时先提取出工具名列表。
        actual_tools = extract_tool_names(result["tool_calls"])
        print("actual tool_calls:", actual_tools)

        # 对比期望工具和实际工具，得到当前题是否通过。
        passed = is_pass(item["expected_tools"], actual_tools)
        print("passed:", passed)

        if passed:
            passed_count += 1

    # 防止 questions.json 为空时出现除零错误。
    pass_rate = passed_count / total if total > 0 else 0

    print("\n---------------------------------")
    print("评估完成")
    print("总题数：", total)
    print("通过数：", passed_count)
    print(f"通过率：{pass_rate:.2%}")
