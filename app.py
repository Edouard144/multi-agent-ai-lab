
from crewai import Agent, Task, Crew, LLM  # LLM 
import os
from dotenv import load_dotenv


# Loading environment variables
load_dotenv()


# Setting up the LLM
llm = LLM(
    model="groq/llama-3.3-70b-versatile",   # ← "provider/model" format
    temperature=0.5
)



# Agent 1: Researcher
researcher = Agent(
    role="Researcher",
    goal="Find key information about a given topic",
    backstory="Expert at finding and summarizing info",
    llm=llm 
)



# Agent 2: Writer
writer = Agent(
    role="Writer",
    goal="Write a clear summary based on research",
    backstory="Turns research into simple content",
    llm=llm
)



# Tasks
research_task = Task(
    description="Research the topic: benefits of AI in education",
    expected_output="5 key points",
    agent=researcher
)

write_task = Task(
    description="Write a short paragraph from the research",
    expected_output="3 sentence summary",
    agent=writer
)




# Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task]
)



# Running the crew
result = crew.kickoff()
print(result)
