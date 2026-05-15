"""模型初始化：DeepSeek 聊天模型 + HuggingFace Embedding 模型。"""

import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# 导入 config 的同时会执行 load_dotenv，确保 .env 已加载。
import config  # noqa: F401


# ChatOpenAI 虽然名字里有 OpenAI，但它支持 OpenAI 兼容接口。
# DeepSeek API 兼容 OpenAI 格式，所以这里可以用 ChatOpenAI 调 DeepSeek。
llm = ChatOpenAI(
    # 优先读取 .env 里的 DEEPSEEK_MODEL；没有就用默认模型。
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    # 读取 DeepSeek API Key。
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    # 读取 DeepSeek API 地址；没有就用官方默认地址。
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    # 关闭 DeepSeek thinking 模式。
    # 原因：当前 demo 使用 LangChain 的 tool calling 流程，不处理 reasoning_content 回传。
    # 如果不关闭，第二轮调用模型时可能出现 reasoning_content 相关 400 报错。
    extra_body={"thinking": {"type": "disabled"}},
)


# Embedding 模型负责把文本转成向量。
# 向量可以用来做语义相似度检索，比如把“RAG 流程”和“文档加载、切分、向量化”匹配起来。
embeddings = HuggingFaceEmbeddings(
    # bge-m3 适合中英文检索。
    model_name="BAAI/bge-m3",
    # 使用 CPU。如果以后有 GPU，可以改成 {"device": "cuda"}。
    model_kwargs={"device": "cpu"},
    # 归一化向量，让相似度计算更稳定。
    encode_kwargs={"normalize_embeddings": True},
)
