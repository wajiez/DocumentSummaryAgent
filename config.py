import redis
from langchain_deepseek import ChatDeepSeek
import os
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

# redis
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# llm
llm = ChatDeepSeek(
    model=os.getenv("ROUTER_MODEL", "deepseek-chat"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    temperature=0,
)

# vect
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",  # 推荐使用 v3 版本，效果更好
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")