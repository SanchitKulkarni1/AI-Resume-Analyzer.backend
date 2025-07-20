import os
import re
import json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Your custom module for creating the vector store
from extract_embed import embed_resume

# Load environment variables from a .env file
load_dotenv()

# Initialize the Language Model
llm = ChatOpenAI(
    model_name="mistralai/mistral-7b-instruct",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)


async def parse_resume(text: str) -> str:
    """
    Asynchronously parses resume text to extract structured data using an LLM.
    Uses LCEL for the chain and .ainvoke() for the non-blocking call.
    """
    prompt = PromptTemplate.from_template(
        """
        You are a resume parser. Extract structured data from the following resume text.
        Return ONLY a single, valid JSON object with the following keys:
        - name
        - email
        - phone
        - education (list of objects with 'degree', 'institution', 'year')
        - work_experience (list of objects with 'role', 'company', 'duration', 'description')
        - skills (list of strings)
        - certifications (list of strings, if any)
        - projects (list of objects with 'name', 'description')
        - links (list of strings for LinkedIn, GitHub, Portfolio etc.)

        Do not include any explanatory text, markdown formatting, or anything before or after the JSON object.

        Resume:
        {resume_text}
        """
    )

    chain = prompt | llm | StrOutputParser()
    parsed_data_content = await chain.ainvoke({"resume_text": text})
    return parsed_data_content


async def analyze_resume(text: str, job_description: str) -> dict:
    """
    Asynchronously analyzes a resume against a job description using a RAG pipeline.
    Uses a modern LCEL-based chain for retrieval and analysis.
    """
    vectordb = await embed_resume(text)
    retriever = vectordb.as_retriever()

    template = """
    You are an expert HR recruiter. You are evaluating a candidate's resume against a job description. Your goal is to fairly assess how well the candidate matches the job and suggest improvements.

    Please analyze the **resume** and **job description** provided below and return a **single valid JSON object** with the following keys and values:

    ---

    **Keys:**

    - "strengths": A list of 3–5 bullet points highlighting strong aspects of the candidate (skills, experience, projects, tools).
    - "improvements": A list of 3–5 bullet points suggesting specific areas for improvement to increase job match.
    - "matching_qualifications": A 1–2 sentence string summarizing which qualifications, experience, or tools match the job.
    - "missing_requirements": A 1–2 sentence string summarizing missing or weak areas relevant to the job description.
    - "score": An integer from 0 to 100 representing how well the resume aligns with the job (based on the rubric below).
    - "final_assessment": A 1–2 sentence recommendation (e.g., whether to shortlist, interview, or advise reskilling).

    ---

    **Scoring Rubric (0–100):**

    - **0–30**: Poor match — lacks most must-have skills, unrelated experience
    - **31–60**: Moderate match — some alignment but missing key qualifications or depth
    - **61–80**: Strong match — has most must-have skills, moderate alignment, potential to grow
    - **81–100**: Excellent match — meets or exceeds all key expectations, ready for next steps

    ---

    Ensure the response is a valid JSON object only — no explanation, no markdown, no extra text.

    Resume:
    {resume}

    Job Description:
    {job_description}
    """

    prompt = PromptTemplate.from_template(template)

    chain = (
        {"resume": RunnablePassthrough(), "job_description": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    raw_text_output = await chain.ainvoke({
        "resume": text,
        "job_description": job_description
    })

    try:
        match = re.search(r"\{.*\}", raw_text_output, re.DOTALL)
        if not match:
            raise json.JSONDecodeError("No JSON object found in LLM output.", raw_text_output, 0)

        clean_json_string = match.group(0)
        analysis_dict = json.loads(clean_json_string)

        return {
            "analysis": {
                "strengths": analysis_dict.get("strengths", []),
                "improvements": analysis_dict.get("improvements", []),
                "matching_qualifications": analysis_dict.get("matching_qualifications", ""),
                "missing_requirements": analysis_dict.get("missing_requirements", ""),
                "final_assessment": analysis_dict.get("final_assessment", "")
            },
            "score": analysis_dict.get("score", 0)
        }

    except json.JSONDecodeError:
        return {
            "error": "Failed to parse structured analysis from LLM.",
            "raw_output": raw_text_output
        }
