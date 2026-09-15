""" Reduce合并文件 """
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from state import AgentState
from config import llm

def reduce_summary(state: AgentState):

    map_results = state.get("map_results", [])
    
    if not map_results:
        print("⚠️ 警告：没有收集到任何分块总结，结束 Reduce。")
        return {
            "document_status": "failed",
            "error": "Map节点未产出有效数据，任务已强制终止。"
        }

    # 加上序号可以让大模型更好地理解文档的逻辑顺序
    formatted_results = "\n\n".join(
        [f"[章节 {i+1} 总结]: {summary}" for i, summary in enumerate(map_results)]
    )
    
    print(f"📑 正在合并 {len(map_results)} 个分块总结，生成最终报告...")

    prompt = ChatPromptTemplate.from_template(
    "你是一个资深的文档分析专家。下面提供了一份长文档各个章节的【分块总结】。\n"
    "请你将这些分块总结进行逻辑重组、去重和润色，生成一份结构清晰、连贯的【最终总结报告】。\n"
    "要求：保留所有关键数据和核心结论，不要遗漏重要信息，语言要专业精炼。\n\n"
    "【分块总结列表】：\n{map_results}\n\n"
    "【最终总结报告】："
    )

    response = llm.invoke(
        prompt.format(map_results = formatted_results)
    )

    final_summary = response.content

    print("✅ 最终总结报告生成完毕！")
    # print(final_summary)
    
    return {
        "extracted_summary": final_summary,
        "document_status": "completed",
        "messages": [AIMessage(content=final_summary)]
    }