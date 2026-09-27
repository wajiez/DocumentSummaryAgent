""" Map摘要文件 """

import json

from langchain_core.prompts import ChatPromptTemplate
from state import AgentState

from config import redis_client, llm

def map_summarize(state: AgentState): #这个节点会被并发执行多次，每次接收不同的 state
    chunk_index = state.get("current_chunk_index")
    document_id = state.get("document_id")

    if not document_id or chunk_index is None:
        print(f"⚠️ 警告：Map节点缺少必要参数，跳过执行。")
        return {"map_results": []}

    redis_key = f"doc_chunks:{document_id}"
    chunks_json = redis_client.get(redis_key) # 取 chunks json
    if not chunks_json:
        print(f"❌ 错误：Redis 中未找到文档块 (Key: {redis_key})")
        return {"map_results": ["错误：未找到文档内容"]}
    chunks = json.loads(chunks_json)
    current_chunk = chunks[chunk_index] # 当前处理块文本

    print(f"🤖 正在总结第 {chunk_index + 1}/{len(chunks)} 块...")

    prompt = ChatPromptTemplate.from_template(
        "你是一个专业的文档分析师。请对以下文档片段进行精炼总结，提取核心事实、数据和关键结论：\n\n"
        "文档片段：\n{chunk}\n\n"
        "精炼总结："
    )
    response = llm.invoke(prompt.format(chunk=current_chunk))
    summary_text = response.content
    
    # LangGraph 会自动将这个单元素的列表追加到全局的 map_results 中
    return {"map_results": [summary_text]}