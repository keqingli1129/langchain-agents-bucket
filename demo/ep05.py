import os,dotenv,sqlite3
from langchain.agents import AgentState,create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import (
    HumanMessage, 
    AIMessage, 
    SystemMessage,
    ToolMessage,
    RemoveMessage
)

from langchain.agents.middleware import before_model,SummarizationMiddleware
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.tools import tool,ToolRuntime
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

os.environ['DEEPSEEK_API_KEY']=dotenv.get_key(".env", "DEEPSEEK_API_KEY")
dsmodel=init_chat_model(model="deepseek-chat")

# LangChain 短期记忆

# 修剪消息中间件
# @before_model
# def cut_messages(state:AgentState, runtime:Runtime):
#     """修剪消息，只保留第一条和最后两条消息"""
#     msgs = state['messages']
#     if len(msgs) < 3:
#         return None  # 消息数量不足，不需要修剪
    
#     first_msg = msgs[0]
#     last_msgs = msgs[-2:]  # 最后两条消息
#     # 正确组合：将第一条消息和最后两条消息组合成列表
#     new_messages = [first_msg] + last_msgs
    
#     return {
#         "messages": [
#             RemoveMessage(id=REMOVE_ALL_MESSAGES),  # 先移除所有消息
#             *new_messages  # 然后添加修剪后的消息
#         ]
#     }

# # 删除指定的消息
# @before_model
# def remove_ok_message(
#     state: AgentState,
#     runtime: Runtime,
# ) -> dict | None:
#     messages = state.get("messages", [])

#     for msg in messages:
#         if msg.type == "human" and msg.content == "OK":
#             # 按 id 删除一条消息
#             return {
#                 "messages": [
#                     RemoveMessage(id=msg.id)
#                 ]
#             }

#     return None

# # # # 摘要内置中间件
# sm_cut=SummarizationMiddleware(
#     model="deepseek-chat",
#     trigger=("tokens",100),
#     keep=("messages",3)
# )


# agent=create_agent(
#     model=dsmodel,
#     middleware=[sm_cut],
#     tools=[],
#     checkpointer=MemorySaver(),
#     system_prompt="你是一个善于推理的小助手。"
#     )

# prompts=[
#     "我是小明的哥哥，我叫小黑。",
#     "我的哥哥是小白。",
#     "OK",
#     "请问小白是小明的什么人？"
# ]

# for idx, msg in enumerate(prompts, 1):
#     response=agent.invoke(
#         {"messages":[HumanMessage(msg)]},
#         {"configurable":{"thread_id":100}}
#         )
    
#     print(f"[{idx}]", "-" * 50)
#     print(response)

# exit()

# 使用工具读取和修改短期记忆中的内容

# # 扩展 AgentState
# class my_agent_state(AgentState):
#     user_id:str
#     role_id:str
#     nick_name:str

# @tool("showuserinfo", description="获取当前智能体使用者的信息", return_direct=True)
# def show_user_info(runtime: ToolRuntime):
#     """显示当前智能体使用者的信息"""
#     user_id = runtime.state.get("user_id", "未知")
#     role_id = runtime.state.get("role_id", "未知")
#     nick_name=runtime.state.get("nick_name","未知")
#     thread_id=runtime.config['configurable'].get("thread_id","未定义")
#     userinfo = f"当前用户id: {user_id}, 角色id: {role_id},线程id{thread_id},昵称为{nick_name}"
#     return userinfo

# @tool("updateuserinfo",description="根据用户要求，修改用户昵称")
# def update_user_nickename(nickname:str,runtime:ToolRuntime):
#     return Command(update={
#         "nick_name":nickname,
#         "messages":[
#             ToolMessage(
#                 content=f"已经根据用户需求将昵称修改成了{nickname},请显示用户修改后的信息",
#                 tool_call_id=runtime.tool_call_id
#                 )
#         ]
#     })


# magent=create_agent(
#     model=dsmodel,
#     middleware=[],
#     tools=[show_user_info,update_user_nickename],
#     checkpointer=MemorySaver(),
#     state_schema=my_agent_state,
#     system_prompt="你是一个善于推理的小助手。能够简洁明了的回答用户的问题。"
#     )

# response=magent.invoke(
#     {
#         # "messages":[HumanMessage("当前使用这个智能体的用户是谁？")],
#         "messages":[HumanMessage("请将我的昵称修改为 凤凰AI七社,并查询修改后的用户信息")],
#         "user_id":"1001",
#         "role_id":"3",
#         "nick_name":"小黑"
#     },
#     {"configurable":{"thread_id":101}}
#     )
# print(response)
# exit()


# LangChain 短期记忆固化存储sqlite

# conn = sqlite3.connect("memory-test.db", check_same_thread=False)
# sqlite_saver = SqliteSaver(conn)
# magent=create_agent(
#     model=dsmodel,
#     middleware=[],
#     tools=[],
#     checkpointer=sqlite_saver,
#     system_prompt="你是一个善于推理的小助手。能够简洁明了的回答用户的问题。"
#     )

# prompts=[
#     "我是小明的哥哥，我叫小黑。",
#     "我的哥哥是小白。",
#     "请问小白是小明的什么人？"
# ]

# for idx, msg in enumerate(prompts, 1):
#     response=magent.invoke(
#         {"messages":[HumanMessage(msg)]},
#         {"configurable":{"thread_id":100}}
#         )
# print(response)

# exit()


# LangChain长期记忆

from langchain_community.embeddings import ZhipuAIEmbeddings
from langgraph.store.sqlite import SqliteStore

os.environ["ZHIPUAI_API_KEY"] = dotenv.get_key(dotenv.find_dotenv(), "ZHIPUAI_API_KEY")
embeddings = ZhipuAIEmbeddings(model="embedding-3")

conn = sqlite3.connect("memory-test.db", check_same_thread=False, isolation_level=None)
store=SqliteStore(
    conn=conn,
    index={"embed": embeddings.embed_documents, "dims": 1024},
)

namespace = ("users","baseinfo")
store.put(namespace, "user001", {"name": "张三", "age": 20, "gender": "男","email":"zhangsan@qq.com"})
store.put(namespace, "user002", {"name": "李四", "age": 20, "gender": "男","email":"lisi@outlook.com"})
store.put(namespace, "user003", {"name": "王五", "age": 20, "gender": "女","email":"wangwu@qq.com"})

# 精确查找
# item = store.get(namespace, "user001")
# print(item)
# exit()
# 语义搜索
# items = store.search(namespace, filter={"gender": "男"}, query="使用qq邮箱的用户", limit=3)
# print(items)
# exit()
# items = store.search(namespace,  query="使用qq邮箱的用户", limit=3)
# print(items)
# exit()
# print(items)

@tool(description="根据用户id获取用户信息")
def get_user_byid(userid:str,runtime:ToolRuntime):
    user_id=runtime.store.context.user_id
    userinfo=runtime.store.get(("users","baseinfo"),user_id)
    runtime.store.put(
        ("users","baseinfo"),
        "user004", 
        {"name": "王六", "age": 40, "gender": "女","email":"wangwu@qq.com"}
        )
    return userinfo

lm_agent=create_agent(
    model=dsmodel,
    middleware=[],
    tools=[get_user_byid],
    store=store,
    system_prompt="你是一个善于推理的小助手。能够简洁明了的回答用户的问题。"
    )