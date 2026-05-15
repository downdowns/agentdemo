import os
import shutil  # 用来删除整个 chroma_db 文件夹
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# 0. 基础配置
# ============================================================

# 加载 .env 文件。
# 这样代码里就可以通过 os.getenv("变量名") 读取 .env 里的配置。
load_dotenv(dotenv_path=".env", override=True)

# 原始知识库文件夹：你自己写的 .md / .txt 文件放这里
DOCS_DIR = "./docs"

# Chroma 向量数据库保存目录：程序会把文档向量保存到这里
CHROMA_DB_DIR = "./chroma_db"

# Chroma 里的 collection 名字，可以理解成“表名”
COLLECTION_NAME = "rag_demo_collection"

# 用来记录 docs 文件夹状态的文件。
# 程序通过它判断：docs 里的文件有没有被修改过。
DOCS_SIGNATURE_FILE = "./chroma_db/docs_signature.txt"


# ============================================================
# 1. 初始化大模型 DeepSeek
# ============================================================

# ChatOpenAI 这个类虽然名字里有 OpenAI，
# 但它支持 OpenAI 兼容接口。
# DeepSeek API 兼容 OpenAI 格式，所以可以用 ChatOpenAI 调用 DeepSeek。
llm = ChatOpenAI(
    # 优先读取 .env 里的 DEEPSEEK_MODEL；如果没有，就使用 deepseek-v4-flash
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    # 读取你的 DeepSeek API Key
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    # 读取 DeepSeek API 地址；如果 .env 没写，就用官方默认地址
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)


# ============================================================
# 2. 初始化 Embedding 模型
# ============================================================

# Embedding 模型不是用来聊天的。
# 它的作用是：把文本变成向量，也就是一串数字。
# 向量可以用来计算语义相似度，比如：
# “RAG 的流程是什么？” 和 “RAG 包括文档加载、切分、向量化...” 会被判断为相关。
embeddings = HuggingFaceEmbeddings(
    # BAAI/bge-m3 是一个常用的 embedding 模型，适合中英文检索
    model_name="BAAI/bge-m3",
    # 使用 CPU 运行。如果你以后有 GPU，可以改成 {"device": "cuda"}
    model_kwargs={"device": "cpu"},
    # normalize_embeddings=True 可以让相似度计算更稳定
    encode_kwargs={"normalize_embeddings": True},
)


# ============================================================
# 3. 读取 docs 文件夹里的本地文档
# ============================================================

def load_local_docs(docs_dir: str):
    """
    读取 docs 文件夹里的 .txt / .md 文件，
    并把每个文件包装成 LangChain 的 Document 对象。

    Document 主要包含两部分：
    1. page_content：文档正文
    2. metadata：额外信息，比如这个内容来自哪个文件
    """
    docs = []

    # 如果 docs 文件夹不存在，直接报错，提醒用户创建
    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"没有找到文档文件夹：{docs_dir}")

    # sorted 是为了让读取顺序固定，方便调试
    for filename in sorted(os.listdir(docs_dir)):
        # 只读取 .txt 和 .md 文件，其他文件忽略
        if not filename.endswith((".txt", ".md")):
            continue

        # 拼出完整路径，例如：./docs/profile.md
        file_path = os.path.join(docs_dir, filename)

        # 读取文件内容
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 如果文件是空的，就跳过，避免空文档进入知识库
        if not content.strip():
            continue

        # 把文件内容包装成 Document
        docs.append(
            Document(
                page_content=content,
                metadata={"source": filename},
            )
        )

    return docs


# ============================================================
# 4. 计算 docs 文件夹状态，用来判断是否需要重建向量库
# ============================================================

def get_docs_signature(docs_dir: str):
    """
    计算 docs 文件夹当前状态。

    这里记录每个 .txt / .md 文件的：
    1. 文件名
    2. 最后修改时间
    3. 文件大小

    只要你修改了 docs 里的文件，
    这个 signature 通常就会变化。
    """
    signature_items = []

    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith((".txt", ".md")):
            continue

        file_path = os.path.join(docs_dir, filename)

        # 文件最后修改时间
        modified_time = os.path.getmtime(file_path)

        # 文件大小，单位是字节
        file_size = os.path.getsize(file_path)

        # 把每个文件的状态拼成一行字符串
        signature_items.append(f"{filename}:{modified_time}:{file_size}")

    # 多个文件状态合并成一个大字符串
    return "\n".join(signature_items)


def load_old_signature(signature_file: str):
    """
    读取上一次构建 Chroma 数据库时保存的 docs 状态。

    如果这个文件不存在，说明：
    1. 可能是第一次运行
    2. 或者以前没有保存过状态
    这时返回 None。
    """
    if not os.path.exists(signature_file):
        return None

    with open(signature_file, "r", encoding="utf-8") as f:
        return f.read()


def save_signature(signature_file: str, signature: str):
    """
    保存当前 docs 文件夹状态。

    当 Chroma 数据库重新构建完成后，
    程序会把当前 docs 状态写入 docs_signature.txt。
    下次启动时，就可以拿它和新的状态做对比。
    """
    # 确保 chroma_db 文件夹存在
    os.makedirs(os.path.dirname(signature_file), exist_ok=True)

    with open(signature_file, "w", encoding="utf-8") as f:
        f.write(signature)


# ============================================================
# 5. 加载文档并切分成小片段
# ============================================================

# 读取 docs 文件夹里的所有文档
docs = load_local_docs(DOCS_DIR)

# 如果 docs 里没有有效文档，直接报错提示
if not docs:
    raise ValueError("docs 文件夹里没有可用的 .txt 或 .md 文档，请先添加内容。")

# 文本切分器：把长文档切成小片段，方便后面做向量检索
text_splitter = RecursiveCharacterTextSplitter(
    # 每个片段大约 100 个字符
    chunk_size=100,
    # 相邻片段之间重叠 20 个字符，避免一句话被切断后丢失上下文
    chunk_overlap=20,
)

# 把原始 Document 切成更小的 Document 片段
splits = text_splitter.split_documents(docs)


# ============================================================
# 6. 创建或加载 Chroma 向量数据库
# ============================================================

# 当前 docs 文件夹状态
current_signature = get_docs_signature(DOCS_DIR)

# 上一次构建向量库时保存的 docs 状态
old_signature = load_old_signature(DOCS_SIGNATURE_FILE)

# 如果两次状态不一样，说明 docs 文件可能被新增、删除或修改过，需要重建向量库
need_rebuild = current_signature != old_signature

# 如果 docs 变了，并且旧的 chroma_db 存在，就删除旧数据库
if need_rebuild and os.path.exists(CHROMA_DB_DIR):
    print("检测到 docs 文件已变化，正在删除旧的 Chroma 向量数据库...")
    shutil.rmtree(CHROMA_DB_DIR)

# 如果 chroma_db 还存在，说明 docs 没变，可以直接加载旧数据库
if os.path.exists(CHROMA_DB_DIR):
    print("检测到已有 Chroma 向量数据库，直接加载中...")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
else:
    # 如果 chroma_db 不存在，就从文档片段重新创建向量数据库
    print("未检测到 Chroma 数据库，开始创建...")
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_DIR,
    )

    # 创建完成后，保存当前 docs 状态
    save_signature(DOCS_SIGNATURE_FILE, current_signature)

def search_docs(query: str, k: int = 2):
    """
    根据用户问题，从 Chroma 向量数据库中检索相关文档片段
    """
    retrieved_docs = vector_store.similarity_search(query, k = k)

    results = []

    for doc in retrieved_docs:
        results.append(
            {
                "source": doc.metadata.get("source", "unknown"),
                "content": doc.page_content
            }
        )
    
    return results

# schema
search_docs_schema = {
    "type": "function",
    "function":{
        "name": "search_docs",
        "description": "搜索本地知识库，适合回答和 RAG、Agent、Function Calling、个人资料相关的问题。",
        "parameters": {
            "type": "object",
            "properties":{
                "query": {
                    "type": "string",
                    "description":"用户的问题或要搜索的关键词"
                },
                "k":{
                    "type":"integer",
                    "description":"要返回的相关文档片段数量，通常取2或3"
                }
            },
            "required": ["query"]
        }
    }
}

# 工具映射
available_functions = {
    "search_docs": search_docs,
}

# ============================================================
# 7. 主程序：交互式 RAG 问答
# ============================================================

if __name__ == "__main__":
    print("\nRAG Agent Demo 初始化成功")
    print("Chat 模型：", llm.model_name)
    print("Embedding 模型：BAAI/bge-m3")
    print("Embedding 类型：HuggingFaceEmbeddings")
    print("原始文档数量：", len(docs))
    print("切分后的文档片段数量：", len(splits))

    # 打印切分后的文档片段，方便你观察知识库到底被切成了什么样
    for i, split in enumerate(splits):
        print(f"\n--- 文档片段 {i + 1} ---")
        print("来源：", split.metadata)
        print("内容：", split.page_content)

    print("\nChroma 向量数据库准备完成")

    # while True 表示一直循环提问，直到用户输入 exit / quit / q
    while True:
        # 第 1 步：接收用户问题
        query = input("\n请输入你的问题，输入 exit 退出：")

        # 如果用户输入 exit / quit / q，就退出程序
        if query.lower() in ["exit", "quit", "q"]:
            print("程序已退出")
            break

        # 如果用户直接回车，不输入内容，就重新开始下一轮
        if not query.strip():
            print("问题不能为空，请重新输入。")
            continue

        # 第 2 步：根据用户问题，从 Chroma 中检索最相关的文档片段
        # k=2 表示取最相关的 2 个片段
        retrieved_results = search_docs(query, k=2)

        print("\n用户问题：", query)
        print("检索到的相关文档数量：", len(retrieved_results))

        # 打印检索结果，方便你判断：模型回答前到底参考了哪些资料
        for i, item in enumerate(retrieved_results):
            print(f"\n--- 检索结果 {i + 1} ---")
            print("来源：", item["source"])
            print("内容：", item["content"])

        # 第 3 步：把检索到的文档片段拼成上下文
        context = "\n\n".join([item["content"] for item in retrieved_results])

        # 第 4 步：构造 prompt，把“上下文 + 用户问题”一起发给 DeepSeek
        prompt = f"""
请你只根据下面的上下文回答用户问题。
如果上下文里没有答案，请直接说：根据当前知识库无法回答。

上下文：
{context}

用户问题：
{query}
"""

        # 第 5 步：调用 DeepSeek 生成最终回答
        answer = llm.invoke(prompt)

        # 第 6 步：打印模型回答
        print("\n模型回答：")
        print(answer.content)
