from crewai import Agent, Task, Crew, LLM
import os
import sys
from dotenv import load_dotenv



# Loading the API key from .env
load_dotenv()



# STEP 1: extracting the file from pdf ot docx
def extract_text_from_file(filepath: str) -> str:
    """Reads a PDF or DOCX file and returns plain text."""

    ext = os.path.splitext(filepath)[1].lower()


    # PDF 
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            os.system("pip install pypdf --quiet")
            from pypdf import PdfReader

        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""  # extracting text from each page
        return text.strip()


    #  DOCX 
    elif ext == ".docx":
        try:
            import docx
        except ImportError:
            os.system("pip install python-docx --quiet")
            import docx

        doc = docx.Document(filepath)
        text = "\n".join([para.text for para in doc.paragraphs])  # put together all paragraphs
        return text.strip()


    #  reject all other formats
    else:
        raise ValueError(f"Unsupported file type: {ext}. Please use .pdf or .docx")




# STEP 2: setting up the llm

llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.5
)



# STEP 3: Creating the agents 


extractor = Agent(
    role="Resume Extractor",
    goal="Extract all key information from the resume text",
    backstory="Expert HR specialist who reads and structures resume data accurately.",
    llm=llm,
    verbose=False
)

evaluator = Agent(
    role="Resume Evaluator",
    goal="Evaluate the strengths and weaknesses of the resume",
    backstory="Senior career coach with 15 years reviewing resumes across all industries.",
    llm=llm,
    verbose=False
)

advisor = Agent(
    role="Career Advisor",
    goal="Give actionable tips to improve the resume",
    backstory="Professional resume writer who helps candidates land jobs at top companies.",
    llm=llm,
    verbose=False
)




# STEP 4: The function that help analysing

def analyse_resume(resume_text: str):
    """Runs 3 agents on the resume text and returns full analysis."""

    # Task 1: Extract info
    extract_task = Task(
        description=f"""
        Read this resume and extract:
        - Full name
        - Contact info (email, phone, location)
        - Work experience (company, role, duration)
        - Education
        - Skills
        - Certifications or achievements

        Resume:
        {resume_text}
        """,
        expected_output="Structured summary of the resume content",
        agent=extractor
    )

    # Task 2: Evaluate quality
    evaluate_task = Task(
        description="""
        Based on the extracted resume, evaluate:
        - Overall strength score (1-10)
        - Top 3 strengths
        - Top 3 weaknesses or missing elements
        - How competitive this candidate looks
        """,
        expected_output="Evaluation with score, strengths, and weaknesses",
        agent=evaluator
    )

    # Task 3: Give improvement tips
    advise_task = Task(
        description="""
        Based on the evaluation, provide:
        - 5 specific actionable improvement tips
        - Skills or certifications to add
        - How to rewrite the summary/objective section
        """,
        expected_output="5 improvement tips with examples",
        agent=advisor
    )
    
    # Task 4: Making a Crew
    crew = Crew(
        agents=[extractor, evaluator, advisor],
        tasks=[extract_task, evaluate_task, advise_task],
        verbose=False
    )

    return crew.kickoff()




# STEP 5: MAIN —--- accepts all specified file path as argument or asks for it

if __name__ == "__main__":

    # Get file path from command line OR prompt the user
    if len(sys.argv) > 1:
        filepath = sys.argv[1] 
    else:
        filepath = input("Enter the path to your CV (.pdf or .docx): ").strip()

    # Check the file exists
    if not os.path.exists(filepath):
        print(f" File not found: {filepath}")
        sys.exit(1)

    print(f"\n Reading file: {filepath}")

    # Extract text from the file
    resume_text = extract_text_from_file(filepath)

    if not resume_text:
        print("Could not extract text from file. Is it a scanned image PDF?")
        sys.exit(1)

    print(" Text extracted successfully!")
    print(f" Characters found: {len(resume_text)}\n")

    # Run the analysis
    print("="*50)
    print("       RESUME ANALYSIS REPORT")
    print("="*50 + "\n")

    result = analyse_resume(resume_text)
    print(result)