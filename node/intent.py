""" 意图路由节点 """

from pydantic import BaseModel, Field
from typing import Literal
import re

from state import AgentState
from config import llm
from langchain_core.prompts import ChatPromptTemplate


class IntentClassification(BaseModel):
    """ 意图分类的结构化输出 """
    intent_type: Literal["summarize", "follow_up", "chat"] = Field(
        description = "用户意图类型: summarize(总结文档),follow_up(针对已有文档追问),chat(无关闲聊)"
    )
    reasoning: str = Field(
        description = "简短的思考过程，解释为什么做出这个分类"
    )

def classify_intent(state:AgentState):
    # ========= test ===========
    if state.get("raw_context"):
        return {
            "next_node": "parse_node"
        }
    
    user_messages = state["messages"][-1].content # 用户最新输入

    # 解析文件路径
    document_url = None
    path_pattern = r'[A-Za-z]:\\[^"]+\.pdf|[A-Za-z]:\\[^"]+\.docx|/[^"]+\.pdf|/[^"]+\.docx'
    match = re.search(path_pattern, user_messages)
    if match:
        document_url = match.group(0)
        document_url = document_url.replace('\\t', '\\t').replace('\\n', '\\n')

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个智能文档处理助手的意图识别模块。
请根据用户的输入以及当前文档的处理状态，判断用户的真实意图。
- 如果用户要求总结、分析、提炼某篇文档，返回 'summarize'。
- 如果文档已经总结过（或用户明确在询问某篇文档的细节），返回 'follow_up'。
- 如果用户只是在打招呼、问天气或进行无关闲聊，返回 'chat'。"""),
        ("human", "当前文档状态: {doc_status}\n用户输入: {user_input}") # 占位符
    ])

    structured_llm = llm.with_structured_output(IntentClassification)

    classification = structured_llm.invoke(
        prompt.format(doc_status = state.get("document_status", "idle"),
        user_input = user_messages)
    )

    intent_to_node = {
        "chat": "chat_node",
        "follow_up": "follow-up_node",
        "summarize": "parse_node"
    }

    next_node = intent_to_node.get(classification.intent_type, "chat")

    print(f"🤔意图识别结果: {classification.intent_type}(原因: {classification.reasoning})")
    print(f"👉下一步进入 {next_node}")

    return {
        "next_node": next_node,
        "document_url": document_url
        }