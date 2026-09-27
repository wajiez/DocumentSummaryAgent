from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from node.intent import classify_intent
from node.chat import chat
from node.parse import parse_document
from node.map_summarize import map_summarize
from node.reduce_summarize import reduce_summary
from node.follow_up import follow_up
from state import AgentState

def intent_node(state: AgentState):
    """ 意图路由 """
    print("💡正在分析用户意图")
    result = classify_intent(state)

    return result

def chat_node(state: AgentState):
    """ 非业务聊天 """
    print("💡正在生成回复")
    result = chat(state)

    return result

def follow_up_node(state: AgentState):
    """ 对文档细节的追问 """
    print("💡正在生成回答")
    result = follow_up(state)

    return result

def parse_node(state: AgentState):
    """ 解析文档 """
    print("💡正在解析文档")
    result = parse_document(state)

    return result

def map_summarize_node(state: dir):
    """ 单块总结 """
    print("💡正在分块总结")
    result = map_summarize(state)
    
    return result

def reduce_summarize_node(state: AgentState):
    """ 合并 """
    print("💡正在合并")
    result = reduce_summary(state)

    return result

# def circuit_break_check_node(state: AgentState):

#     return 
g = StateGraph(AgentState)

g.add_node("intent_node", intent_node)
g.add_node("chat_node", chat_node)
g.add_node("follow-up_node", follow_up_node)
g.add_node("parse_node", parse_node)
g.add_node("map_summarize_node", map_summarize_node)
g.add_node("reduce_summarize_node", reduce_summarize_node)

def route_by_intent(state: AgentState):
    """ 根据 next_node 字段的值决定下一个节点 """
    next_step = state.get("next_node")
    if next_step == "parse_node":
        return "parse_node"
    if next_step == "follow-up_node":
        return "follow-up_node"
    if next_step == "chat_node":
        return "chat_node"
    else:
        return "chat_node"

g.add_edge(START, "intent_node")
g.add_conditional_edges("intent_node",
                        route_by_intent,
                        {
                            "parse_node": "parse_node",
                            "follow-up_node": "follow-up_node",
                            "chat_node": "chat_node"
                        })
g.add_edge("chat_node", END)
g.add_edge("follow-up_node", END)

def route_after_parse(state: AgentState):
    if state.get("document_status") == "failed" or state.get("error"):
            print("🛑 检测到任务失败，正在终止对话流...")
            return "end_graph"
    return [
        Send("map_summarize_node", {
            "current_chunk_index": i,
            "document_id": state.get("document_id")
        }) 
        for i in range(state.get("total_chunks"))
    ]

g.add_conditional_edges(
    "parse_node",    
    route_after_parse,                
    {
        "end_graph":END
    }
)

g.add_edge("map_summarize_node", "reduce_summarize_node")

def route_after_reduce(state: AgentState):
    if state.get("document_status") == "failed" or state.get("error"):
        print("🛑 检测到任务失败，正在终止对话流...")
        return "end_graph"
    return "continue"

g.add_conditional_edges(
    "reduce_summarize_node",
    route_after_reduce,
    {
        "end_graph":END,
        "continue": END
    }
)

memory = MemorySaver()
g = g.compile(checkpointer=memory)

def run():
    # 会话配置
    config = {"configurable": {"thread_id": "user_12345"}}
    
    # 用户初始输入
    user_input = input("👤 用户: ")
    
    # 多轮对话循环
    while True:
        if user_input.lower() in ["exit", "quit", "退出"]:
            print("👋 再见！")
            break
            
        # 触发图执行（传入 config，LangGraph 会自动读取历史记忆）
        result = g.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config
        )
        
        if result.get("document_status") == "failed":
            error = result.get("error")
            print(f"🤖 助手: 任务处理失败，{error}。\n")
            user_input = input("👤 用户: ")
            continue 

        # 打印 AI 的最新回复
        ai_message = result["messages"][-1].content
        print(f"🤖 助手: {ai_message}\n")
        
        # 等待用户的下一次输入
        user_input = input("👤 用户: ")
    # g.invoke({
    #     "messages": [{"role": "user", "content": "帮我总结一下这篇论文"}],
    #     "document_url": r"D:\githubProject\document summarize  agent\test doc\Clustering_based_Autoencoder_for_Anomaly_Detection1.pdf"
    # })

if __name__ == "__main__":
    run()
