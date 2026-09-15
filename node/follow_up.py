
from state import AgentState
from langchain_core.prompts import ChatPromptTemplate

from config import llm, vectorstore

def follow_up(state: AgentState):
    user_query = state["messages"][-1].content
    extracted_summary = state.get("extracted_summary", "暂无总结报告")
    document_id = state.get("document_id")

    retrieved_chunks = []
    if document_id:
        docs = vectorstore.similarity_search(
            query=user_query,
            k=3,
            filter={"document_id": document_id} 
        )
        retrieved_chunks = [doc.page_content for doc in docs]
        
    formatted_chunks = "\n\n".join(retrieved_chunks) if retrieved_chunks else "未检索到相关原文片段"
    
    print(f"🔍 正在基于向量检索回答追问: '{user_query[:30]}...'")

    prompt = ChatPromptTemplate.from_template(
        "你是一个专业的文档分析助手。请严格根据以下【检索到的原文片段】和【总结报告】回答用户的【追问】。\n"
        "如果提供的信息不足以回答问题，请诚实回答不知道，绝对不要编造。\n\n"
        "【总结报告】：\n{extracted_summary}\n\n"
        "【检索到的原文片段】：\n{retrieved_chunks}\n\n"
        "【用户追问】：\n{user_query}\n\n"
        "【你的回答】："
    )

    response = llm.invoke(prompt.format(
        extracted_summary=extracted_summary,
        retrieved_chunks=formatted_chunks,
        user_query=user_query
    ))
    
    # 将回答追加到全局消息列表
    return {"messages": [response]}