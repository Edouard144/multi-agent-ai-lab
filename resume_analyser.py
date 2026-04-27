# resume_analyser.py
from crewai import Agent, Task, Crew, LLM
import os
from dotenv import load_dotenv



# Load API key from .env file
load_dotenv()



# Setting up the LLM 
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.5
)



# AGENTS — Each has a specific role in analysing the resume


# Agent 1: Reads and understands the resume
extractor = Agent(
    role="Resume Extractor",
    goal="Extract all key information from the resume text",
    backstory="""You are an expert HR specialist who reads resumes 
    and pulls out structured information clearly and accurately.""",
    llm=llm,
    verbose=False
)

# Agent 2: Evaluates strengths and weaknesses
evaluator = Agent(
    role="Resume Evaluator",
    goal="Evaluate the strengths and weaknesses of the resume",
    backstory="""You are a senior career coach with 15 years of experience 
    reviewing resumes across tech, business, and creative fields.""",
    llm=llm,
    verbose=False
)

# Agent 3: Gives improvement suggestions
advisor = Agent(
    role="Career Advisor",
    goal="Give actionable tips to improve the resume",
    backstory="""You are a professional resume writer who helps 
    candidates land jobs at top companies by improving their resumes.""",
    llm=llm,
    verbose=False
)




# FUNCTION — Takes resume text and runs the full analysis


def analyse_resume(resume_text: str):

    # TASKS 

    # Task 1: Extract key info from the resume
    extract_task = Task(
        description=f"""
        Read the following resume and extract:
        - Full name
        - Contact info (email, phone, location)
        - Work experience (company, role, duration)
        - Education
        - Skills
        - Certifications or achievements (if any)

        Resume:
        {resume_text}
        """,
        expected_output="A clean structured summary of the resume content",
        agent=extractor
    )

    # Task 2: Evaluate the resume quality
    evaluate_task = Task(
        description="""
        Based on the extracted resume information, evaluate:
        - Overall resume strength (score it 1-10)
        - Top 3 strengths
        - Top 3 weaknesses or missing elements
        - How competitive this candidate looks
        """,
        expected_output="A clear evaluation with score, strengths, and weaknesses",
        agent=evaluator
    )

    # Task 3: Give specific improvement tips
    advise_task = Task(
        description="""
        Based on the evaluation, give:
        - 5 specific and actionable tips to improve the resume
        - Suggest what skills or certifications to add
        - Recommend how to rewrite the summary/objective section
        """,
        expected_output="5 clear improvement tips with examples where possible",
        agent=advisor
    )




    # CREW 

    crew = Crew(
        agents=[extractor, evaluator, advisor],
        tasks=[extract_task, evaluate_task, advise_task],
        verbose=False
    )


    # Run the full analysis
    result = crew.kickoff()
    return result


# MAIN — Paste your resume text here


if __name__ == "__main__":

    #Paste your resume text between the triple quotes (""")
    my_resume = """
    John Doe
    Email: johndoe@email.com | Phone: +1 234 567 890 | Location: New York, USA

    SUMMARY
    Software developer with 3 years of experience in Python and web development.
    Passionate about building scalable applications.

    EXPERIENCE
    Junior Developer — TechCorp (2021 - 2023)
    - Built REST APIs using Flask
    - Worked on frontend with React
    - Collaborated with a team of 5 developers

    Intern — StartupXYZ (2020 - 2021)
    - Assisted in debugging and testing
    - Wrote documentation

    EDUCATION
    B.Sc. Computer Science — State University (2017 - 2021)

    SKILLS
    Python, Flask, React, Git, SQL

    CERTIFICATIONS
    None
    """

    print("\n" + "="*50)
    print("       RESUME ANALYSIS REPORT")
    print("="*50 + "\n")

    result = analyse_resume(my_resume)
    print(result)