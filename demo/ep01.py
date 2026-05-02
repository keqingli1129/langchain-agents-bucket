import os,dotenv
from langchain.agents import create_agent
os.environ['DEEPSEEK_API_KEY']=dotenv.get_key(".env", "DEEPSEEK_API_KEY")
first_agent=create_agent(
    model="deepseek-chat",
    tools=[],
    middleware=[],
    system_prompt="You are a helpful assistant that provides weather information."
    )
response=first_agent.invoke({"messages": [{"role": "user", "content": "今天天气怎么样？"}]})
print(response)