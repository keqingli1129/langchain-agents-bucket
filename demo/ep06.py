import os,dotenv,sqlite3
from langchain.agents import create_agent
from langgraph.runtime import Runtime
from langchain.messages import HumanMessage, ToolMessage
from langgraph.types import Command
from typing import Callable
from langchain.tools.tool_node import ToolCallRequest
# 从 langchain.agents.middleware 导入所有中间件类
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    HumanInTheLoopMiddleware,
    PIIMiddleware,  # PII Detection
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    SummarizationMiddleware,
    ContextEditingMiddleware,
    TodoListMiddleware,
    LLMToolSelectorMiddleware,
    LLMToolEmulator,
    ShellToolMiddleware,
    FilesystemFileSearchMiddleware,
    ClearToolUsesEdit,
    HostExecutionPolicy,

    before_model,
    after_agent,
    after_model,
    before_agent,
    wrap_model_call,
    wrap_tool_call,
    AgentState,
    ModelRequest,
    ModelResponse,
)
os.environ['DEEPSEEK_API_KEY']=dotenv.get_key(".env", "DEEPSEEK_API_KEY")
# =======================================================
# LangChain 内置中间件
# =======================================================

# # 1. 模型调用限制：防止死循环和高额账单
# model_limit = ModelCallLimitMiddleware(
#     run_limit=10,         
#     thread_limit=100,     
#     exit_behavior="error" # 'error'(报错), 'end'(直接结束), 'continue'(忽略)
# )

# # 2. 工具调用限制：防止 API 滥用
# tool_limit = ToolCallLimitMiddleware(
#     tool_name="web_search",
#     thread_limit=100,  
#     run_limit=3,          
#     exit_behavior="end"     # 'error'(报错), 'end'(直接结束), 'continue'(忽略)
# )

# # 3. 人类介入/审批：关键操作需人工确认
# hitl=HumanInTheLoopMiddleware(
#     interrupt_on={
#         "sendemail":True,
#         "readfile":{"allowed_decisions": ["approve", "reject"]},
#         "deletefile":False
#     },
#     description_prefix="请问是否允许以下操作："
# )

# # 4. 敏感信息检测 (PII)：保护隐私
# pii_detection =  PIIMiddleware(
#     "email", 
#     strategy="block", #redact mask hash
#     apply_to_input=True,#检查用户输入
#     # apply_to_input=False,# 不检查模型输出
#     apply_to_tool_results=True #检查工具的返回信息
# )



# # 5. 模型回退：主模型挂了切备用
# model_fallback = ModelFallbackMiddleware(
#     "deepseek-v3",
#     "gpt-3.5-turbo", 
#     "claude-3-haiku"
# )

# # 6. 模型重试：网络抖动自动重试
# model_retry = ModelRetryMiddleware(
#     max_retries=3,
#     initial_delay=1,         
#     backoff_factor=2,     
#     on_failure="continue" # error
# )

# # 7. 工具重试：API 报错自动重试
# tool_retry = ToolRetryMiddleware(
#     max_retries=3,
#     backoff_factor=2.0,
#     initial_delay=1.0,
#     on_failure="continue", # error
#     tools=[]
# )

# # 8. 自动总结 
# summarization = SummarizationMiddleware(
#     model="deepseek-chat",
#     trigger=[("tokens", 4000), ("messages",20)],
#     # trigger={"tokens": 4000, "messages": 20},
#     keep={"messages": 5}
# )

# # 9. 上下文编辑
# context_editing = ContextEditingMiddleware(
#             edits=[
#                 ClearToolUsesEdit(
#                     trigger=100000,
#                     keep=3,
#                 ),
#             ],
#         ),

# # 10. 待办事项
# todo = TodoListMiddleware()  # 自动注入 manage_todos 工具

# # 11. LLM 工具选择器
# selector = LLMToolSelectorMiddleware(
#     model="gpt-3.5-turbo", # 使用轻量模型做筛选
#     max_tools=5,            # 即使有 50 个工具，只选最相关的 5 个给主 Agent
#     always_include=["search"]
# )

# # 12. 工具模拟器 (测试用)
# emulator = LLMToolEmulator(
#     model='deepseek-chat',
#     tools=["expensive_api_tool"] # None,[],拦截这些工具的调用
# )

# # 13. Shell 工具注入
# shell = ShellToolMiddleware(
#     workspace_root="./sandbox",    # 限制访问目录
#     execution_policy=HostExecutionPolicy(), # 执行的安全策略
# )

# # 14. 文件搜索注入
# file_search = FilesystemFileSearchMiddleware(
#     root_path="./docs",
#     max_file_size_mb=2
# )


#自定义中间件

@before_agent
def befagent(state: AgentState, runtime: Runtime):
    """代理主流程开始前触发"""
    print("智能体调用之前")

@before_model
def befmodel(state: AgentState, runtime: Runtime):
    """模型调用前触发"""
    print("模型调用之前")
    

@after_model
def endmodel(state: AgentState, runtime: Runtime):
    """模型调用完成后触发"""
    print("模型执行结束")

@after_agent()
def endagent(state: AgentState, runtime: Runtime):
    """代理整体流程完成后触发"""
    print("智能体执行结束")



cache={}
@wrap_model_call
def cache_wrapper(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],):
    key = hash(str(request.messages))
    if key in cache:
        return cache[key]

    response = handler(request)
    cache[key] = response
    return response


@wrap_tool_call
def weather_guard(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:

    tool_name = request.tool_call["name"]
    args = request.tool_call["args"]

    if tool_name == "get_weather":
        city = args.get("city")

        if city != "北京":
            # ❌ 不执行 handler → 工具不会被调用
            return ToolMessage(
                content=f"系统限制：暂不支持查询 {city} 的天气",
                tool_call_id=request.tool_call["id"],
            )

    # ✅ 放行，执行真实工具
    return handler(request)


agent = create_agent(
    model="deepseek-chat",
    tools=[],
    middleware=[
        endagent,
        endmodel,
        befagent,
        befmodel,
        # cache_wrapper,
        # weather_guard
    ]
)
agent.invoke({"messages":[HumanMessage("hi")]})