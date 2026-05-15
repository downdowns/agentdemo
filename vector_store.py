"""文档加载、文本切分、Chroma 向量库创建/加载。"""

import os
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHROMA_DB_DIR, COLLECTION_NAME, DOCS_DIR, DOCS_SIGNATURE_FILE
from models import embeddings


def load_local_docs(docs_dir: str) -> list[Document]:
    """
    读取 docs 文件夹里的 .txt / .md 文件，并包装成 LangChain Document。

    Document 包含：
    - page_content：正文内容
    - metadata：元数据，比如来源文件名
    """
    # 先判断 docs 文件夹是否存在。
    # 如果不存在，后面的 os.listdir(docs_dir) 会报更难懂的错误，
    # 所以这里主动抛出一个更明确的错误。
    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"没有找到文档文件夹：{docs_dir}")

    # 创建一个空列表，用来存放读取出来的 Document 对象。
    docs: list[Document] = []

    # 遍历 docs 文件夹下的所有文件名。
    # sorted(...) 是为了固定读取顺序，方便调试和复现结果。
    for filename in sorted(os.listdir(docs_dir)):
        # 只读取 txt 和 md，其它文件忽略。
        if not filename.endswith((".txt", ".md")):
            continue

        # 拼接完整文件路径，例如：./docs/profile.md。
        file_path = os.path.join(docs_dir, filename)

        # 以 UTF-8 编码读取文件内容。
        # with open(...) 可以保证文件读取结束后自动关闭。
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 跳过空文件，避免空内容进入知识库。
        if not content.strip():
            continue

        # 把普通字符串包装成 LangChain 的 Document 对象。
        # page_content 存正文，metadata 存来源文件名。
        docs.append(
            Document(
                page_content=content,
                metadata={"source": filename},
            )
        )

    # 返回读取到的所有文档。
    return docs


def get_docs_signature(docs_dir: str) -> str:
    """
    计算 docs 文件夹的“状态签名”。

    签名由以下信息组成：
    - 文件名
    - 文件最后修改时间
    - 文件大小

    只要你新增、删除或修改 docs 里的 .md / .txt 文件，签名通常就会变化。
    """
    # 用列表保存每个文件的状态信息。
    signature_items: list[str] = []

    # 遍历 docs 目录下所有文件，顺序固定，避免签名顺序随机变化。
    for filename in sorted(os.listdir(docs_dir)):
        # 只统计 txt / md 文件，其它文件不影响知识库。
        if not filename.endswith((".txt", ".md")):
            continue

        # 拼接完整路径。
        file_path = os.path.join(docs_dir, filename)

        # 获取文件最后修改时间。
        modified_time = os.path.getmtime(file_path)

        # 获取文件大小，单位是字节。
        file_size = os.path.getsize(file_path)

        # 把一个文件的状态拼成字符串，加入列表。
        signature_items.append(f"{filename}:{modified_time}:{file_size}")

    # 把所有文件状态合并成一个字符串，作为整个 docs 文件夹的签名。
    return "\n".join(signature_items)


def load_old_signature(signature_file: str) -> str | None:
    """读取上一次构建向量库时保存的 docs 状态签名。"""
    # 如果签名文件不存在，说明可能是第一次运行。
    if not os.path.exists(signature_file):
        return None

    # 读取并返回旧签名内容。
    with open(signature_file, "r", encoding="utf-8") as f:
        return f.read()


def save_signature(signature_file: str, signature: str) -> None:
    """保存当前 docs 状态签名，用于下次运行时判断是否需要重建向量库。"""
    # 先确保签名文件所在的文件夹存在。
    # exist_ok=True 表示：如果文件夹已存在，不要报错。
    os.makedirs(os.path.dirname(signature_file), exist_ok=True)

    # 把当前签名写入文件，覆盖旧内容。
    with open(signature_file, "w", encoding="utf-8") as f:
        f.write(signature)


def split_docs(docs: list[Document]) -> list[Document]:
    """把长文档切分成更小的片段，方便向量检索。"""
    text_splitter = RecursiveCharacterTextSplitter(
        # 每个片段大约 100 个字符。
        chunk_size=100,
        # 相邻片段重叠 20 个字符，避免上下文被切断。
        chunk_overlap=20,
    )

    # 执行切分，返回切分后的 Document 片段列表。
    return text_splitter.split_documents(docs)


def load_or_create_vector_store(splits: list[Document]) -> Chroma:
    """
    加载或创建 Chroma 向量数据库。

    逻辑：
    1. 计算当前 docs 文件状态
    2. 和上次保存的状态对比
    3. 如果 docs 变了，删除旧 Chroma 数据库并重建
    4. 如果 docs 没变，直接加载已有 Chroma 数据库
    """
    # 计算当前 docs 文件夹状态。
    current_signature = get_docs_signature(DOCS_DIR)

    # 读取上一次构建 Chroma 时保存的 docs 状态。
    old_signature = load_old_signature(DOCS_SIGNATURE_FILE)

    # 如果当前状态和旧状态不同，说明 docs 文件有变化，需要重建向量库。
    need_rebuild = current_signature != old_signature

    # 如果需要重建，并且旧向量库存在，就先删除旧向量库目录。
    if need_rebuild and os.path.exists(CHROMA_DB_DIR):
        print("检测到 docs 文件已变化，正在删除旧的 Chroma 向量数据库...")
        shutil.rmtree(CHROMA_DB_DIR)

    # 如果 Chroma 目录存在，说明可以直接加载已有向量库。
    if os.path.exists(CHROMA_DB_DIR):
        print("检测到已有 Chroma 向量数据库，直接加载中...")
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_DIR,
        )

    # 如果走到这里，说明没有可用的 Chroma，需要从文档片段重新创建。
    print("未检测到 Chroma 数据库，开始创建...")
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_DIR,
    )

    # 保存当前 docs 状态，方便下次启动时判断是否变化。
    save_signature(DOCS_SIGNATURE_FILE, current_signature)

    # 返回创建好的向量数据库对象。
    return vector_store


# 读取原始文档。
docs = load_local_docs(DOCS_DIR)

# 如果没有任何有效文档，程序无法做 RAG，直接提示用户。
if not docs:
    raise ValueError("docs 文件夹里没有可用的 .txt 或 .md 文档，请先添加内容。")

# 把原始文档切分成小片段。
splits = split_docs(docs)

# 加载或创建 Chroma 向量数据库。
vector_store = load_or_create_vector_store(splits)
