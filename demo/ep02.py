import os,dotenv
from datetime import datetime
from langchain.agents import create_agent
from langchain.tools import tool,ToolRuntime
from pydantic import BaseModel, Field
from langchain.agents.structured_output import ToolStrategy
from zoneinfo import ZoneInfo

os.environ['DEEPSEEK_API_KEY']=dotenv.get_key(".env", "DEEPSEEK_API_KEY")

@tool
def get_time():
    """获取当前的北京时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class getCityTimeInput(BaseModel):
    '''时间工具参数'''
    city:str=Field(description="需要获取时间的城市中文名称")

@tool(
    "get_city_time",
    description="获取指定地点的当前时间",
    return_direct=True,
    args_schema=getCityTimeInput
    )
def get_city_time(city:str):
    city_to_tz = {
        '伦敦': 'Europe/London',
        '纽约': 'America/New_York',
        '东京': 'Asia/Tokyo',
        '上海': 'Asia/Shanghai',
        '悉尼': 'Australia/Sydney'
    }
    
    tz_name = city_to_tz.get(city)
    if not tz_name:
        return f"抱歉，暂不支持查询 {city} 的时间。支持的城市：{', '.join(city_to_tz.keys())}"
    
    try:
        citytime = datetime.now(ZoneInfo(tz_name))
        return f"{city}的当前时间是：{citytime.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    except Exception as e:
        return f"获取 {city} 时间时出错：{str(e)}"

#智能体结构化输出
@tool(
    "get_peom_detail",
    description="使用给出的古诗名，返回古诗的详细信息",
    return_direct=True
    )
def peom_detail(pname:str):
    return pname+"作者未知"
class pomeModel(BaseModel):
    """诗歌详情"""
    title: str = Field(description="诗歌标题")
    auth: str = Field(description="诗歌作者")
    period: str = Field(description="年代")
    content: str = Field(description="诗歌全文")
    story: str = Field(description="这首诗歌的故事背景")


first_agent=create_agent(
    model="deepseek-chat",
    tools=[get_time,get_city_time,peom_detail],
    system_prompt="你是一个小秘书，会在需要调用工具的时候，调用工具回答问题。",
    response_format=ToolStrategy(pomeModel),
    debug=True
    )

import logging
logging.basicConfig(level=logging.DEBUG)

response=first_agent.invoke(
    {"messages":[{"role":"user","content":"请问当前北京是什么时间？"}]},
    )

# response=first_agent.invoke(
#     {"messages":[{"role":"user","content":"整理古诗《将进酒》的详细信息"}]}
#     )

print(response)

