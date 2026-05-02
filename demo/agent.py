import os,dotenv
from datetime import datetime
from langchain.agents import create_agent,AgentState
from langgraph.runtime import Runtime
from langchain.tools import tool,ToolRuntime
from langchain.messages import AIMessage, HumanMessage
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    SummarizationMiddleware,
    after_model,
)
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware

os.environ['DEEPSEEK_API_KEY']=dotenv.get_key(".env", "DEEPSEEK_API_KEY")

# 自定义工具
@tool(description="儿童动画片推荐工具")
def recommend_anime(runtime:ToolRuntime)->str:
    return "《圣斗士星矢》"

@tool(description="根据文件名称删除本地文件")
def delete_file(filename:str):
    return "文件已经删除"

@tool(description="根据文件名称读取文件内容")
def read_file(filename:str):
    return "文件内容已经读取成功"

# 内置中间件

summarization = SummarizationMiddleware(
    model="deepseek-chat",
    trigger=[("tokens", 200), ("messages",20)],
    keep=("messages", 5)
)

hitl= HumanInTheLoopMiddleware(
    interrupt_on={
        "delete_file":True,
        "read_file":False,
        "recommend_anime":{"allowed_decisions": ["approve", "reject"]},
    },
    description_prefix="请问是否允许以下操作："
)

#自定义中间件
@after_model
def endmodel(state: AgentState, runtime: Runtime):
    """
    模型调用完成后触发：
    给大模型回复的结果增加一个时间戳
    如果消息包含 tool_calls，不添加时间戳
    """
    messages = state.get("messages", [])
    if not messages:
        return None
    
    # 获取最后一条消息
    last_message = messages[-1]
    
    # 检查是否是 AI 消息且有内容，且不包含 tool_calls
    if isinstance(last_message, AIMessage) and last_message.content:
        # 如果消息包含 tool_calls，不修改它
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return None
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 创建新的消息内容，添加时间戳
        original_content = last_message.content
        new_content = f"{original_content}\n\n[时间戳: {timestamp}]"
        
        # 创建新的 AIMessage，不包含 tool_calls
        new_message = AIMessage(
            content=new_content,
            id=last_message.id,
            response_metadata=last_message.response_metadata if hasattr(last_message, 'response_metadata') else {},
            usage_metadata=last_message.usage_metadata if hasattr(last_message, 'usage_metadata') else {},
        )
        
        # 返回更新后的消息列表
        updated_messages = messages[:-1] + [new_message]
        return {"messages": updated_messages}
    
    return None

agent = create_agent(
    model="deepseek-chat",
    tools=[
        recommend_anime,
        delete_file,
        read_file
        ],
    middleware=[
        summarization,
        hitl,
        endmodel
    ],
    system_prompt="你是一个资深的儿童教育专家。"
)
