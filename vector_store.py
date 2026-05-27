"""文档加载、文本切分、Chroma 向量库创建/加载。"""

import csv
import os
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHROMA_DB_DIR, COLLECTION_NAME, DOCS_DIR, DOCS_SIGNATURE_FILE
from models import embeddings


def read_csv_as_text(file_path: str) -> str:
    """
    读取 CSV 文件，并转换成适合 RAG 检索的纯文本。

    为什么不直接 f.read()？
    - CSV 是结构化数据，有表头和多行记录；
    - 直接读也能读到文本，但语义不够清晰；
    - 转成“字段名：字段值”的格式，更利于 embedding 和检索。

    示例输出：
    第 1 条记录：
    category：supplier_onboarding
    question：供应商入驻需要准备哪些资料？
    answer：需要营业执照...
    """
    rows_text: list[str] = []

    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row_index, row in enumerate(reader, start=1):
            row_lines = [f"第 {row_index} 条记录："]

            for column_name, value in row.items():
                clean_value = (value or "").strip()
                if clean_value:
                    row_lines.append(f"{column_name}：{clean_value}")

            rows_text.append("\n".join(row_lines))

    return "\n\n".join(rows_text)

def load_csv_docs(file_path: str, filename: str) -> list[Document]:
    """
    读取 CSV 文件，并把每一行转换成一个独立的 LangChain Document。

    为什么要每行一个 Document？
    - CSV 通常是一行一条结构化知识，例如一条 FAQ、一条商品规则、一条配置记录；
    - 每行独立成 Document 后，检索粒度更细；
    - metadata 可以记录 row_index、category、question 等字段；
    - 后续做 bad case 分析时，可以定位到 CSV 的具体哪一行。
    """
    csv_docs: list[Document] = []

    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row_index, row in enumerate(reader, start=1):
            row_lines = [f"第 {row_index} 条记录："]

            for column_name, value in row.items():
                clean_value = (value or "").strip()
                if clean_value:
                    row_lines.append(f"{column_name}：{clean_value}")

            page_content = "\n".join(row_lines)

            if not page_content.strip():
                continue

            metadata = {
                "source": filename,
                "row_index": row_index,
            }

            # 如果 CSV 里有这些字段，就顺手放进 metadata。
            # 这样后续检索、日志、分析时能更精确定位。
            for metadata_field in ["category", "question"]:
                field_value = (row.get(metadata_field) or "").strip()
                if field_value:
                    metadata[metadata_field] = field_value

            csv_docs.append(
                Document(
                    page_content=page_content,
                    metadata=metadata,
                )
            )

    return csv_docs

def load_local_docs(docs_dir: str) -> list[Document]:
    """
    读取 docs 文件夹里的 .txt / .md / .csv 文件，并包装成 LangChain Document。

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
        # 只读取 txt / md / csv，其它文件暂时忽略。
        if not filename.endswith((".txt", ".md", ".csv")):
            continue

        # 拼接完整文件路径，例如：./docs/profile.md。
        file_path = os.path.join(docs_dir, filename)

        # 根据文件类型选择不同读取方式。
        # - md / txt：本身就是纯文本，直接读取；
        # - csv：结构化表格数据，每一行转换成一个独立 Document；
        if filename.endswith(".csv"):
            # append: 把整个列表作为一个元素塞进去
            # extend：把列表里的每个元素逐个塞进去
            docs.extend(load_csv_docs(file_path, filename))
            continue
        else:
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

    只要你新增、删除或修改 docs 里的 .md / .txt / .csv 文件，签名通常就会变化。
    """
    # 用列表保存每个文件的状态信息。
    signature_items: list[str] = []

    # 遍历 docs 目录下所有文件，顺序固定，避免签名顺序随机变化。
    for filename in sorted(os.listdir(docs_dir)):
        # 只统计 txt / md / csv 文件，其它文件不影响知识库。
        if not filename.endswith((".txt", ".md", ".csv")):
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
    """
    把原始 Document 切分成更小的 chunk，并为每个 chunk 补充评估和溯源 metadata。

    输入：
    - docs：load_local_docs() 读取出来的原始 Document 列表。
      - md / txt：通常是“一个文件 = 一个 Document”
      - csv：当前是“CSV 每一行 = 一个 Document”

    输出：
    - splits：切分后的 chunk 列表，每个 chunk 仍然是一个 Document。

    本函数负责两件事：
    1. 使用 RecursiveCharacterTextSplitter 做文本切分；
    2. 给每个 chunk 补充 chunk_index 和 chunk_id，方便后续检索评估、日志分析和引用定位。
    """
    # 创建文本切分器。
    #
    # RecursiveCharacterTextSplitter 的思路是：
    # 优先按更自然的边界切分文本，例如段落、换行、句子；
    # 如果文本仍然太长，再继续递归地按更小粒度切分。
    #
    # 这样比“每 500 个字符硬切一刀”更容易保留语义完整性。
    text_splitter = RecursiveCharacterTextSplitter(
        # 每个片段大约 500 个字符。
        # 500 左右更适合当前 Markdown 技术文档，能保留较完整的小节语义。
        chunk_size=500,
        # 相邻片段重叠 80 个字符，避免关键上下文被切断。
        # 例如某个概念刚好在 chunk 边界附近，overlap 可以让前后两个 chunk 都保留一部分上下文。
        chunk_overlap=80,
    )

    # 执行真正的切分。
    #
    # split_documents 会保留原始 Document 的 metadata。
    # 例如 CSV 行级 Document 里的 source、row_index、category、question，
    # 在切分后仍然会跟着对应 chunk 走。
    splits = text_splitter.split_documents(docs)

    # 用来记录“每个计数维度”已经生成了多少个 chunk。
    #
    # 为什么不用一个全局计数？
    # - 我们希望每个 source 文件都有自己的 chunk 编号；
    # - 对 CSV 来说，我们希望每一行也有自己的 chunk 编号。
    #
    # 普通 md/txt 的 key：
    #   langchain_rag.md
    #
    # CSV 行级 Document 的 key：
    #   b2b_faq.csv::row_003
    chunk_count_by_key: dict[str, int] = {}

    # 遍历每一个切分后的 chunk，为它补充 chunk_index 和 chunk_id。
    for split in splits:
        # source 表示这个 chunk 来自哪个原始文件。
        # 普通文档示例：langchain_rag.md
        # CSV 文档示例：b2b_faq.csv
        source = split.metadata.get("source", "unknown")

        # row_index 只会出现在 CSV 行级 Document 中。
        # 普通 md/txt 没有 row_index，因此这里会是 None。
        row_index = split.metadata.get("row_index")

        # 决定当前 chunk 应该按哪个维度计数。
        #
        # 对普通 md/txt：
        #   chunk_key = "langchain_rag.md"
        #
        # 对 CSV 第 3 行：
        #   chunk_key = "b2b_faq.csv::row_003"
        #
        # 这样可以保证 CSV 每一行的 chunk 编号都从 0 开始。
        if row_index is not None:
            chunk_key = f"{source}::row_{int(row_index):03d}"
        else:
            chunk_key = source

        # 取出当前 key 已经生成过多少个 chunk。
        # 如果这个 key 第一次出现，就从 0 开始。
        chunk_index = chunk_count_by_key.get(chunk_key, 0)

        # 当前 chunk 用掉了这个编号，所以计数 +1，给下一个 chunk 使用。
        chunk_count_by_key[chunk_key] = chunk_index + 1

        # chunk_index 是数字编号，方便程序排序、统计和展示。
        split.metadata["chunk_index"] = chunk_index

        # chunk_id 是稳定、可读的字符串 ID，用于：
        # - eval/questions.json 里的 expected_chunk_ids；
        # - Chunk Recall@k / MRR@3 等检索指标；
        # - 日志分析和 bad case 定位；
        # - 回答引用和来源追踪。
        #
        # 普通 md/txt 示例：
        #   langchain_rag.md::chunk_003
        #
        # CSV 行级示例：
        #   b2b_faq.csv::row_003::chunk_000
        if row_index is not None:
            split.metadata["chunk_id"] = (
                f"{source}::row_{int(row_index):03d}::chunk_{chunk_index:03d}"
            )
        else:
            split.metadata["chunk_id"] = f"{source}::chunk_{chunk_index:03d}"

    # 返回已经切分好，并且补充了 chunk metadata 的 Document 列表。
    return splits


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
