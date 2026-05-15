"""项目配置文件。"""

from dotenv import load_dotenv


# 加载 .env 文件，这样其它模块可以用 os.getenv("变量名") 读取配置。
load_dotenv(dotenv_path=".env", override=True)

# docs 文件夹：存放你的本地知识库原文，例如 .md / .txt 文件。
DOCS_DIR = "./docs"

# Chroma 数据库文件夹：存放向量化后的知识库。
CHROMA_DB_DIR = "./chroma_db"

# Chroma collection 名字，可以理解成向量数据库里的“表名”。
COLLECTION_NAME = "rag_demo_collection"

# 用来记录 docs 文件状态的文件。
# 程序会用它判断 docs 文件是否发生变化，从而决定要不要重建向量库。
DOCS_SIGNATURE_FILE = "./chroma_db/docs_signature.txt"

# Agent 最大循环轮数，防止模型一直调用工具导致死循环。
MAX_AGENT_ROUNDS = 5
