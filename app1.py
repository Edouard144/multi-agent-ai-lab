from crewai import Agent, Task, Crew
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq



# Loading environment variables
load_dotenv()



# setting up the LLM
llm = ChatGroq(
    model="llama3-8b-8192",
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



# Run the crew
result = crew.kickoff()
print(result)
