import redis
from langchain_deepseek import ChatDeepSeek
import os
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import DashScopeEmbeddings

# redis
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# llm
llm = ChatDeepSeek(
    model=os.getenv("ROUTER_MODEL", "deepseek-chat"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    temperature=0,
)

# embeddings_split = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")

# 本地 模型
embeddings = HuggingFaceEmbeddings(
    model_name="D:/models/bge-large-zh-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)
# spliter
text_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",   # 百分位数法，默认95
    breakpoint_threshold_amount=95,            # 值越小切分越多
    min_chunk_size=100,                        # 最小块大小，避免碎片
)

# vect
# embeddings = DashScopeEmbeddings(
#     model="text-embedding-v3",  # 推荐使用 v3 版本，效果更好
#     dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
# )
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")