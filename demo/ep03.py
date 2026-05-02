import os,dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage,ToolMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import BaseModel, Field
from langchain.tools import tool,ToolRuntime

os.environ['DEEPSEEK_API_KEY']=dotenv.get_key(".env", "DEEPSEEK_API_KEY")

# first_agent=create_agent(
#     model="deepseek-chat",
#     tools=[],
#     middleware=[],
#     system_prompt="不管用户说什么，你都回答：小心天狼星人的间谍……"
#     )

# 精确配置一个model
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1, 
    check_every_n_seconds=0.1,
    max_bucket_size=10,
)

model=init_chat_model(
    model="deepseek-chat",
    # model="deepseek_reasoner",
    # model="google_gemini_2.5_flash",
    # api_key="",
    # base_url="",
    # temperature="0.7",
    # max_tokens="",
    # timeout="",
    # rate_limiter=rate_limiter
)

# agent=create_agent(
#     model=model,
#     system_prompt="你是一个聪明伶俐的小秘书"
#     )

#如何运行一个model
model_msg="你好小秘书"
model_msg=[
    {"role":"system","content":"你是一个聪明伶俐的小秘书"},
    {"role":"user","content":"你好小秘书"}
    ]
modle_msg=[
    SystemMessage(content="你是一个聪明伶俐的小秘书"),
    HumanMessage(content="你好小秘书")
]

# model_response=model.invoke(model_msg)
# print(model_response)

# stream流式输出
# for chunk in model.stream(modle_msg):
#     print(chunk.content,end="",flush=True)

# batch 批量处理
# prompts=["写一首五言律诗","1+1=?小学数学，直接给出答案","给出五个常用的中国人名"];
# reponse=model.batch(prompts)
# print(reponse)

# for chunk in model.batch_as_completed(prompts,config={"max_concurrency":3}):
#     print(chunk,end="",flush=True)

# model工具调用
# @tool(description="儿童动画片推荐工具")
# def recommend_anime()->str:
#     return "《小马宝莉》"

# dsmodelwt=model.bind_tools([recommend_anime,])
# aimsg=dsmodelwt.invoke("请推荐一个动画片")

# # print(aimsg)
# # exit()
# # 检查模型是否调用了工具
# if aimsg.tool_calls:
#     # 使用 工具的.invoke() 方法调用工具
#     toolresult = recommend_anime.invoke({})
#     # 获取第一个工具调用的 ID
#     tool_call_id = aimsg.tool_calls[0]['id']
#     toolmsg = ToolMessage(content=toolresult, tool_call_id=tool_call_id)
    
#     # 将工具结果发送回模型
#     res = dsmodelwt.invoke([
#         *model_msg,
#         aimsg,
#         toolmsg
#     ])
#     print(res)
# else:
#     # 模型没有调用工具，直接输出回复
#     print(aimsg.content)
# agent=create_agent(model=dsmodelwt)
# agent.invoke(model_msg)


# class pomeModel(BaseModel):
#     """诗歌详情"""
#     title: str = Field(description="诗歌标题")
#     auth: str = Field(description="诗歌作者")
#     period: str = Field(description="年代")
#     content: str = Field(description="诗歌全文")
#     story: str = Field(description="这首诗歌的故事背景")

# withstrc = model.with_structured_output(pomeModel)
# response = withstrc.invoke("古诗《将进酒》的详细信息")

# agent=create_agent(model=withstrc)
# print(response)


#运行时可配置模型
cacmodel=init_chat_model()
cacmodel.invoke(
    "你好",
    config={
        "configurable":{"model":"deepseek-chat"}
    }
    )

cacmodel=init_chat_model(
    model="deepseek-chat",
    temperature=0.7,
    configurable_fields={"model","model_provider","temperature"}
    )
    
cacmodel.invoke(
    "你好",
    config={
        "configurable":{"model":"deepseek-reasoner"}
    }
    )