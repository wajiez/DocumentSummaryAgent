from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict, Literal

from langgraph.graph import add_messages
from langgraph.graph.message import AnyMessage

import operator

class AgentState(TypedDict):
    """ golbal state """
    # 外部传入
    document_url: str
    user_input: str

    # 控制
    next_node: Optional[str]
    document_status: Optional[Literal["idle", "parsed", "summarizing", "completed", "failed"]]

    # 中间字段
    document_id: Optional[str] # 内部生成的文档ID
    total_chunks: Optional[str] # 进度追踪
    
    current_chunk_index: Optional[int] 
    # next_steps: Optional[list]
    map_results: Annotated[list[str], operator.add] # 多个map节点并发执行，返回的列表合并
    extracted_summary: Optional[str] # reduce总结

    final_summary:Optional[str]

    # messages
    messages: Annotated[list[AnyMessage], add_messages] # 追加

    # error
    error: Optional[str]

    # =========== test ===========
    raw_context: Optional[str]
    redis_key: Optional[str]

def state_schema() -> dict[str, Any]:
    """导出字段名 → 注解，方便调试时打印状态结构，也方便后续做校验。"""
    return dict(AgentState.__annotations__)