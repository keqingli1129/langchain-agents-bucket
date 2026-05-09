import os
from dotenv import load_dotenv
from langchain.agents import create_agent   

load_dotenv()
first_agent=create_agent(
        model="deepseek-chat",
        tools=[],
        middleware=[],
        system_prompt="you are a good assistant"
    )

def main():
    
    print("Hello from langchain-agents-bucket!")
    
    response=first_agent.invoke(
        {"messages":[{"role":"user", "content":"who are you"}]}
    )
    print(response)


if __name__ == "__main__":
    main()
