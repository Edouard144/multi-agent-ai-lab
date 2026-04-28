from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv


load_dotenv()


# LLM setup
llm = ChatOpenAI(
    model="llama3-8b-8192",
    temperature=0
)


# Tool used to search on google
search_tool = SerperDevTool()



# Agent that use the tool
researcher = Agent(
    role="News Researcher",
    goal="Find latest information from the internet",
    backstory="Expert at finding current news",
    tools=[search_tool],   # 👈 tool attached
    llm=llm
)



# Task to be performed by the created agent above
task = Task(
    description="Find the latest news about artificial intelligence in 2026",
    expected_output="3 short bullet points of recent news",
    agent=researcher
)



# Crew of only one agent
crew = Crew(
    agents=[researcher],
    tasks=[task]
)



# Running the program
result = crew.kickoff()
print(result)