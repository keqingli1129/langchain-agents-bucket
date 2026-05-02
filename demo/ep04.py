import os,dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage,ToolMessage
from langchain.tools import tool,ToolRuntime

os.environ['DEEPSEEK_API_KEY']=dotenv.get_key(".env", "DEEPSEEK_API_KEY")

# LangChain Messages

dsmodel=init_chat_model(model="deepseek-chat")

dsmodel.invoke("小明5岁了，请给他推荐一些动画片")

humanmsg=HumanMessage("小明5岁了，请给他推荐一些动画片")
systemmsg=SystemMessage("你是一个儿童教育专家，能够简单直接的回答我的问题")
# response=dsmodel.invoke([systemmsg,humanmsg])
# print(response)

# AIMessage(
#     content=(
#         "为5岁的小明挑选动画片时，需要兼顾**趣味性、教育意义和适龄性**。这个阶段的孩子正处于好奇心旺盛、"
#         "学习社交规则和情感发展的关键期，以下推荐分为几个类别，方便家长根据小明的兴趣和需求选择：\n\n"
#         "---\n\n"
#         "### 🌟 **温馨成长 & 情商培养类**\n"
#         "1. **《小猪佩奇》**  \n"
#         "2. **《汪汪队立大功》**  \n"
#         "3. **《布鲁伊》**  \n"
#         "   - **特点**：被誉为“育儿神作”，通过狗狗一家的游戏展现亲子互动、想象力和情绪管理，家长也能从中获得启发。\n\n"
#         ……"
#     ),
#     additional_kwargs={
#         "refusal": None
#     },
#     response_metadata={
#         "token_usage": {
#             "completion_tokens": 854,
#             "prompt_tokens": 14,
#             "total_tokens": 868,
#             "completion_tokens_details": None,
#             "prompt_tokens_details": {
#                 "audio_tokens": None,
#                 "cached_tokens": 0
#             },
#             "prompt_cache_hit_tokens": 0,
#             "prompt_cache_miss_tokens": 14
#         },
#         "model_provider": "deepseek",
#         "model_name": "deepseek-chat",
#         "system_fingerprint": "fp_eaab8d114b_prod0820_fp8_kvcache",
#         "id": "ab67a803-e08e-4305-95d8-89aab82a79c6",
#         "finish_reason": "stop",
#         "logprobs": None
#     },
#     usage_metadata={
#         "input_tokens": 14,
#         "output_tokens": 854,
#         "total_tokens": 868,
#         "input_token_details": {
#             "cache_read": 0
#         },
#         "output_token_details": {}
#     },
#     id="lc_run--019b4f2a-c129-7561-a9dd-0cb5ffa858cf-0"
# )
# exit()
@tool(description="儿童动画片推荐工具")
def recommend_anime()->str:
    return "《小马宝莉》"

dsmodelwt=dsmodel.bind_tools([recommend_anime])

aimsg=dsmodelwt.invoke([systemmsg,humanmsg])
print(type(aimsg))
print(aimsg)

# 输出的打印出来的结果，结构是这样的
# AIMessage(
#     content="我来为5岁的小明推荐一些适合的动画片。",
#     additional_kwargs={
#         "refusal": None
#     },
#     response_metadata={
#         "token_usage": {
#             "completion_tokens": 42,
#             "prompt_tokens": 307,
#             "total_tokens": 349,
#             "completion_tokens_details": None,
#             "prompt_tokens_details": {
#                 "audio_tokens": None,
#                 "cached_tokens": 256
#             },
#             "prompt_cache_hit_tokens": 256,
#             "prompt_cache_miss_tokens": 51
#         },
#         "model_provider": "deepseek",
#         "model_name": "deepseek-chat",
#         "system_fingerprint": "fp_eaab8d114b_prod0820_fp8_kvcache",
#         "id": "2ec305d9-915a-4d54-82f4-cbd702ab2f7c",
#         "finish_reason": "tool_calls",
#         "logprobs": None
#     },
#     tool_calls=[
#         {
#             "name": "recommend_anime",
#             "args": {},
#             "id": "call_00_ia012kblhP0d8vjSkAqkgn6u",
#             "type": "tool_call"
#         }
#     ],
#     usage_metadata={
#         "input_tokens": 307,
#         "output_tokens": 42,
#         "total_tokens": 349,
#         "input_token_details": {
#             "cache_read": 256
#         },
#         "output_token_details": {}
#     },
#     id="lc_run--019b4f22-17f8-7cd0-8564-d89f1ac95158-0"
# )
# exit()
# 检查模型是否调用了工具
if aimsg.tool_calls:
    # 使用工具的 .invoke() 方法调用工具
    toolresult = recommend_anime.invoke({})
    # 获取工具调用的 ID
    tool_call_id = aimsg.tool_calls[0]['id']
    toolmsg = ToolMessage(content=toolresult, tool_call_id=tool_call_id)
    
    # 将工具结果发送回模型
    res = dsmodelwt.invoke([
        systemmsg,
        humanmsg,
        aimsg,
        toolmsg
    ])
    print(res)
else:
    # 模型没有调用工具，直接输出回复
    print(aimsg.content)

msglist=[
        systemmsg,
        humanmsg,
        aimsg,
        toolmsg,
        aimsg,
        humanmsg,
        aimsg,
        humanmsg,
        aimsg,
        toolmsg,
        aimsg,
        humanmsg,
        aimsg,
        humanmsg,
        aimsg,
        toolmsg,
        aimsg,
        humanmsg,
        aimsg,
        humanmsg,
        aimsg,
        toolmsg,
        aimsg,
        humanmsg,
        aimsg,
]

#LangChain content_blocks

human_message = HumanMessage(content_blocks=[
    {"type": "text", "text": "请推荐一个适合儿童的动画片"},
    {"type": "image", "url": "https://example.com/image.jpg"},
    {
        "type": "video", 
        "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
        "mime_type": "video/mp4",
    },
    {"type": "audio", "url": "https://example.com/image.mp3"},
    {"type": "file", "url": "https://example.com/file.pdf"},
])