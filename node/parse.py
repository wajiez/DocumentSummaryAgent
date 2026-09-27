""" 解析分块文件 """

import json
import pymupdf
import uuid
from typing import List

from state import AgentState
from config import redis_client, vectorstore
from langgraph.types import Send
from config import text_splitter


def parse_document(state: AgentState):
    # ============ test ===============
    raw = state.get("raw_context")
    if raw is not None:
        docs = text_splitter.create_documents([raw])
        chunks = [d.page_content for d in docs]

        redis_key = f"doc_chunks:{state["redis_key"]}"
        redis_client.set(redis_key, json.dumps(chunks))

        return {
            "document_id": state["redis_key"],
            "total_chunks": len(chunks),
            "document_status": "parsed"
        }
    # =======================================
    else:
        doc_path = state.get("document_url")
        if not doc_path:
            print("⚠️ 警告：没有拿到有效 document_url，结束 Parse。")

            return {
                "document_status": "failed",
                "error": "document_url 无效，任务已强制终止。"
            }

        document_id = str(uuid.uuid4())[:8] # 作为 Redis 的 Key
        
        try:
            doc = pymupdf.open(doc_path)

            chunks: List[str] = []

            for page in doc:
                page_text = page.get_text("text")  # 按阅读顺序返回纯文本

                docs = text_splitter.create_documents([page_text]) # 语义切分
                chunks.extend([d.page_content for d in docs])

            # for page in doc:
            #     page_text = page.get_text("text") # 按阅读顺序返回纯文本
            #     page_chunks = [chunk.strip() for chunk in page_text.split("\n\n") if chunk.strip()] # 按段落分
            #     chunks.extend(page_chunks)

            redis_key = f"doc_chunks:{document_id}"
            redis_client.set(redis_key, json.dumps(chunks))
            print(f"✅ 文档解析完成，共 {len(chunks)} 个块已存入 Redis (Key: {redis_key})")

            metadatas = [{"document_id": document_id, "chunk_index": i} for i in range(len(chunks))]
            vectorstore.add_texts(texts=chunks, metadatas=metadatas)
            
            print(f"✅ 文档已解析并存入向量库 (Document ID: {document_id})")
            
            return {
                "document_id": document_id,
                "total_chunks": len(chunks),
                "document_status": "parsed"
            }
            
        except Exception as e:
            print(f"❌ 文档解析失败: {e}")
            return {"document_status": "failed"}