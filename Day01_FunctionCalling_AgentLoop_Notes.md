# Day 01：Function Calling + Agent Loop 面试笔记

> 今日目标：能清楚讲明白 **Function Calling 工具调用流程**，理解 tool schema、参数校验、异常处理、Agent Loop 终止条件，并能回答常见面试问题。

---

## 目录

1. [一句话总览](#1-一句话总览)
2. [Function Calling 完整流程](#2-function-calling-完整流程)
3. [Tool Schema 的作用](#3-tool-schema-的作用)
4. [参数校验](#4-参数校验)
5. [工具异常处理](#5-工具异常处理)
6. [Agent Loop](#6-agent-loop)
7. [Function Calling vs ReAct](#7-function-calling-vs-react)
8. [常见面试问题](#8-常见面试问题)
9. [极简背诵版](#9-极简背诵版)
10. [今天的复习任务](#10-今天的复习任务)

---

# 1. 一句话总览

Function Calling 的本质：

> **模型负责决定调用哪个工具以及生成参数，程序负责真正执行工具。**

Agent Loop 的本质：

> **循环执行“模型判断是否调用工具 → 程序执行工具 → 工具结果返回模型”，直到模型不再调用工具或达到终止条件。**

---

# 2. Function Calling 完整流程

Function Calling 不是模型真的执行函数，而是模型生成一个结构化的工具调用请求。

完整流程：

```text
用户输入问题
  ↓
程序把用户问题 + tools schema 发给模型
  ↓
模型判断是否需要调用工具
  ↓
如果需要，模型返回 tool_calls
  ↓
程序解析 tool_calls，拿到工具名和参数
  ↓
程序执行对应的 Python 函数
  ↓
程序把工具结果作为 tool message 放回 messages
  ↓
再次调用模型
  ↓
模型基于工具结果生成最终回答
```

示例：

用户问：

```text
15 加 27 等于多少？
```

模型可能返回：

```json
{
  "name": "calculator",
  "args": {
    "operation": "add",
    "a": 15,
    "b": 27
  }
}
```

程序真正执行：

```python
function_response = calculator(operation="add", a=15, b=27)
```

然后把结果返回给模型：

```json
{
  "result": 42
}
```

最终模型回答：

```text
15 加 27 等于 42。
```

面试表达：

> Function Calling 是一种结构化工具调用机制。模型根据用户问题和 tool schema 生成工具名和参数，程序解析后执行真实函数，再把工具结果返回给模型生成最终回答。

---

# 3. Tool Schema 的作用

Tool Schema 是给模型看的“工具说明书”。

它告诉模型：

```text
有哪些工具
每个工具叫什么名字
每个工具适合做什么
工具需要哪些参数
参数类型是什么
哪些参数必填
```

示例：

```python
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
```

## 3.1 Schema 里最重要的字段

| 字段 | 作用 |
|---|---|
| `name` | 工具名称，模型调用工具时会返回这个名字 |
| `description` | 描述工具用途，影响模型什么时候调用它 |
| `parameters` | 定义工具参数结构 |
| `properties` | 每个参数的类型和说明 |
| `required` | 哪些参数必须提供 |
| `enum` | 限制参数只能从固定值中选择 |

## 3.2 Schema 的关键理解

Tool Schema 的作用是 **约束和引导模型输出工具调用格式**。

但它不是万能的。

它不能替代：

```text
服务端参数校验
权限控制
异常处理
业务规则判断
```

面试表达：

> Tool Schema 是模型理解工具能力和参数格式的依据，但它只是约束模型输出，不能替代服务端校验。

---

# 4. 参数校验

模型生成的参数不一定永远可靠，所以程序侧必须做参数校验。

常见问题：

```text
工具名不存在
缺少必填参数
参数类型不对
参数值超出范围
除法时 b = 0
用户没有权限访问某个资源
参数存在安全风险
```

## 4.1 工具名校验

```python
if function_name not in AVAILABLE_FUNCTIONS:
    function_response = {"error": f"未知工具：{function_name}"}
```

## 4.2 参数值校验

```python
if operation not in ["add", "subtract", "multiply", "divide"]:
    return {"error": "Unknown operation"}
```

## 4.3 业务规则校验

```python
if operation == "divide" and b == 0:
    return {"error": "Division by zero"}
```

## 4.4 生产中的校验重点

生产系统中还要检查：

```text
用户是否有权限
参数是否越界
是否访问敏感资源
是否会触发危险操作
是否需要人工确认
```

面试表达：

> Schema 可以减少错误参数，但不能保证绝对正确。生产中必须在服务端做参数校验、权限校验和异常捕获。

---

# 5. 工具异常处理

工具调用可能失败。

常见失败原因：

```text
参数错误
工具不存在
外部 API 超时
数据库连接失败
权限不足
第三方服务限流
工具内部代码异常
```

## 5.1 不推荐的做法

不要让异常直接把 Agent 程序打崩：

```python
function_response = AVAILABLE_FUNCTIONS[function_name](**function_args)
```

如果工具内部报错，整个程序可能中断。

## 5.2 推荐做法

捕获异常，返回结构化错误：

```python
try:
    function_response = AVAILABLE_FUNCTIONS[function_name](**function_args)
except Exception as e:
    function_response = {"error": str(e)}
```

然后把错误结果也返回给模型：

```python
messages.append({
    "role": "tool",
    "tool_call_id": tool_call_id,
    "content": json.dumps(function_response, ensure_ascii=False),
})
```

这样模型可以基于错误结果回答用户：

```text
天气服务暂时不可用，请稍后再试。
```

## 5.3 生产中的增强处理

生产环境可以进一步做：

```text
失败重试
服务降级
缓存兜底
错误日志
trace id 追踪
告警监控
```

面试表达：

> 工具失败后不应该让 Agent 崩溃，而是捕获异常，返回结构化错误给模型，必要时做重试、降级和日志记录。

---

# 6. Agent Loop

## 6.1 Agent Loop 是什么

普通 Function Calling 通常只处理一轮：

```text
模型调用工具
程序执行工具
模型回答
结束
```

Agent Loop 可以多轮执行：

```text
模型调用工具
程序执行工具
模型继续判断是否还要调用工具
如果还要，继续执行
直到模型不再调用工具
输出最终回答
```

适合多步骤任务。

例如：

```text
请先查询 RAG 的核心流程，再计算 15 加 27，最后告诉我上海天气。
```

Agent 可能执行：

```text
第 1 轮：调用 search_docs
第 2 轮：调用 calculator
第 3 轮：调用 get_weather
第 4 轮：最终回答
```

也可能一轮同时调用多个工具。

---

## 6.2 Agent Loop 基础结构

```python
for round_num in range(1, MAX_AGENT_ROUNDS + 1):
    response = llm.invoke(
        messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    if not response.tool_calls:
        print(response.content)
        break

    messages.append(response)

    for tool_call in response.tool_calls:
        execute_tool(tool_call)
        messages.append(tool_result_message)
```

---

## 6.3 Agent Loop 终止条件

### 1. 模型不再调用工具

最正常的终止条件：

```python
if not message.tool_calls:
    print(message.content)
    return
```

含义：

```text
模型已经拿到足够信息，可以生成最终回答。
```

---

### 2. 达到最大轮数

防止模型无限调用工具：

```python
MAX_AGENT_ROUNDS = 5
```

达到最大轮数后停止：

```python
print("达到最大 Agent 轮数，程序停止继续调用工具。")
```

---

### 3. 工具出现不可恢复错误

例如：

```text
权限不足
关键参数缺失
外部服务不可用
```

可以选择终止，或者把错误返回给模型，让模型解释给用户。

---

### 4. 用户主动退出

命令行程序中：

```python
if query.lower() in ["exit", "quit", "q"]:
    break
```

---

## 6.4 为什么有时一轮只调用一个工具

这是模型策略，不是代码错误。

可能原因：

```text
模型更保守
模型想先拿一个工具结果再决定下一步
工具之间可能存在依赖
模型没有选择并行调用
```

只要代码里支持：

```python
for tool_call in message.tool_calls:
```

就说明程序支持一轮多个工具。

面试表达：

> 一轮调用一个工具或多个工具都是正常行为。具体取决于模型策略和任务是否存在依赖关系。

---

# 7. Function Calling vs ReAct

## 7.1 Function Calling

Function Calling 是一种 **结构化工具调用机制**。

模型输出的是结构化的 tool call：

```json
{
  "name": "search_docs",
  "args": {
    "query": "RAG 的核心流程"
  }
}
```

特点：

```text
结构化
稳定
程序容易解析
更适合生产系统
```

---

## 7.2 ReAct

ReAct = Reasoning + Acting。

它是一种 Agent 思考与行动范式。

典型格式：

```text
Thought: 我需要先查询资料
Action: search_docs
Action Input: RAG 的核心流程
Observation: 查询结果...
Thought: 我已经知道答案
Final Answer: ...
```

特点：

```text
强调边思考边行动
更灵活
但文本解析更脆弱
```

---

## 7.3 核心区别

| 对比项 | Function Calling | ReAct |
|---|---|---|
| 本质 | 结构化工具调用协议 | 推理 + 行动的 Agent 范式 |
| 输出格式 | JSON / tool_calls | Thought / Action / Observation 文本 |
| 稳定性 | 更稳定 | 更灵活但解析更脆弱 |
| 工程落地 | 更适合生产 | 更偏 Agent 思维模式 |

面试表达：

> Function Calling 更像模型和程序之间的结构化工具调用协议；ReAct 更像一种让模型边推理边行动的 Agent 思考范式。实际项目中可以用 Function Calling 实现 ReAct 风格的 Agent Loop。

---

# 8. 常见面试问题

## Q1：Function Calling 的原理是什么？

回答：

> Function Calling 的原理是：模型根据用户问题和 tool schema 判断是否需要调用工具。如果需要，模型会生成结构化的 tool_calls，包括工具名和参数。程序解析 tool_calls，执行真实函数，并把工具结果返回给模型，最后由模型生成自然语言回答。

---

## Q2：模型真的会执行函数吗？

回答：

> 不会。模型只会生成工具调用请求，例如工具名和参数。真正执行函数的是业务代码，比如 Python 代码根据工具名从 AVAILABLE_FUNCTIONS 里找到对应函数并执行。

---

## Q3：tool schema 有什么作用？

回答：

> Tool schema 是给模型看的工具说明书，告诉模型工具名称、用途、参数类型和必填字段。它可以引导模型生成正确的工具调用格式，但不能替代服务端参数校验。

---

## Q4：如果模型生成的参数不合法怎么办？

回答：

> 不能完全信任模型参数。程序侧需要检查工具名是否存在、必填参数是否完整、参数类型和值是否合法，还要做权限校验和异常捕获。如果参数不合法，就返回结构化错误给模型。

---

## Q5：如何防止模型乱调用工具？

回答：

> 可以从多层控制：写清楚 tool description，在 system prompt 中规定工具使用规则，用 enum 等方式限制参数范围，程序侧使用工具白名单，并做权限校验。不能只依赖 prompt。

---

## Q6：Function Calling 和 ReAct 有什么区别？

回答：

> Function Calling 是结构化工具调用机制，模型输出 tool_calls，程序容易解析，更适合工程落地。ReAct 是 Reasoning + Acting 的 Agent 范式，强调 Thought、Action、Observation 的循环推理过程。实际项目中可以用 Function Calling 实现类似 ReAct 的 Agent Loop。

---

## Q7：工具调用失败后怎么处理？

回答：

> 工具失败后不应该让程序崩溃。应该捕获异常，返回结构化错误结果，并把错误作为 tool message 返回给模型，让模型告诉用户失败原因。生产中还可以加重试、降级、缓存兜底和日志监控。

---

## Q8：Agent Loop 的终止条件有哪些？

回答：

> 最主要的终止条件是模型不再返回 tool_calls，说明它已经可以最终回答。为了防止死循环，还需要设置最大轮数。其他终止条件包括工具出现不可恢复错误，或者用户主动退出。

---

## Q9：为什么模型有时一轮只调用一个工具？

回答：

> 这是模型的工具调用策略。它可能认为工具之间存在依赖，或者想先拿到一个工具结果再决定下一步。只要代码支持遍历 tool_calls，就能处理一轮多个工具；串行调用多个工具也是正常 Agent 行为。

---

# 9. 极简背诵版

面试前重点背这几句：

1. **模型不执行函数，程序执行函数。**
2. **Function Calling 的本质是：模型生成工具名和参数，程序执行工具，再把结果返回模型。**
3. **Tool Schema 是给模型看的工具说明书，描述工具用途和参数格式。**
4. **Schema 不能代替服务端校验，生产中必须做参数校验、权限校验和异常处理。**
5. **工具失败后要返回结构化错误，不要让 Agent 崩溃。**
6. **Agent Loop 会循环执行：模型请求工具 → 程序执行工具 → 工具结果返回模型。**
7. **Agent Loop 的主要终止条件是：模型不再调用工具，或达到最大轮数。**
8. **Function Calling 是结构化工具调用协议，ReAct 是推理 + 行动的 Agent 范式。**
9. **防止乱调用工具要靠 schema、system prompt、工具白名单、参数约束和权限控制。**
10. **一轮一个工具或一轮多个工具都正常，取决于模型策略和任务依赖关系。**

---

# 10. 今天的复习任务

用自己的话回答下面 5 个问题：

1. Function Calling 的完整流程是什么？
2. Tool Schema 为什么不能替代参数校验？
3. 工具调用失败后应该怎么处理？
4. Agent Loop 什么时候结束？
5. Function Calling 和 ReAct 有什么区别？
