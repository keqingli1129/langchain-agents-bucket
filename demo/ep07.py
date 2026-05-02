import os,dotenv
from langchain.agents import create_agent,AgentState
from langgraph.runtime import Runtime
from langchain.tools import tool,ToolRuntime
from langchain.agents.middleware import (
    SummarizationMiddleware,
    after_model,
)

os.environ['DEEPSEEK_API_KEY']=dotenv.get_key(".env", "DEEPSEEK_API_KEY")

@tool(description="儿童动画片推荐工具")
def recommend_anime(runtime:ToolRuntime)->str:
    return "《圣斗士星矢》"

# 内置中间件
summarization = SummarizationMiddleware(
    model="deepseek-chat",
    trigger=[("tokens", 200), ("messages",20)],
    keep={"messages": 5}
)

#自定义中间件
@after_model
def endmodel(state: AgentState, runtime: Runtime):
    """
    模型调用完成后触发：
    给大模型回复的结果总，增加一个时间戳
    """
    print("模型执行结束")

agent = create_agent(
    model="deepseek-chat",
    tools=[recommend_anime], # 基础工具列表
    middleware=[
        summarization,
        endmodel
    ],
    system_prompt="你是一个资深的儿童教育专家。"
)