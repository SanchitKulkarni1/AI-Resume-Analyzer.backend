from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from extract_embed import embed_resume
import os
import re
import json


# Load environment variables
load_dotenv()

# Initialize LLM using OpenRouter's free model
llm = ChatOpenAI(
    model_name="mistralai/mistral-7b-instruct",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

# Resume parser to extract structured JSON data from text
def parse_resume(text):
    prompt = PromptTemplate.from_template("""
        You are a resume parser. Extract structured data from the following resume text and return it in JSON format with keys:
        - name
        - email
        - phone
        - education (list of degrees, institutions, years)
        - work_experience (list of roles, companies, durations)
        - skills (list of technical and soft skills)
        - certifications (if any)
        - projects (name + short description)
        - links (LinkedIn, GitHub, Portfolio etc.)

        Resume:
        {resume_text}
    """)

    final_prompt = prompt.format(resume_text=text)
    parsed_data = llm.invoke(final_prompt)


    return parsed_data.content #since we are normal and not QA Rerieval

# Resume analysis against job description
def analyze_resume(text, job_description):
    vectordb = embed_resume(text)
    retriever = vectordb.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    # ==> STEP 1: Update the prompt to ask for a JSON output <==
    prompt = PromptTemplate.from_template("""
        You are an expert HR recruiter. Analyze the resume against the job description.
        Return a single, valid JSON object with the following keys:
        "strengths": A list of strings highlighting the candidate's strong points.
        "improvements": A list of strings for areas of improvement.
        "matching_qualifications": A single string summarizing the matching qualifications.
        "missing_requirements": A single string summarizing any missing requirements.
        "score": An integer score from 0-100 representing the match.
        "final_assessment": A single string with your final recommendation.

        **Job Description:**
        {job_description}

        **Resume Content:**
        {context}
    """)

    final_prompt = prompt.format(job_description=job_description, context=text)
    
    # ==> STEP 2: Get the raw string result from the chain <==
    analysis_result = qa.invoke(final_prompt)
    raw_text_output = analysis_result['result']

    # ==> STEP 3: Clean and parse the string into a dictionary <==
    try:
        # Use regex to find the JSON blob, just in case the LLM adds extra text
        match = re.search(r"\{.*\}", raw_text_output, re.DOTALL)
        if match:
            clean_json_string = match.group(0)
            # Convert the clean JSON string into a Python dictionary
            analysis_dict = json.loads(clean_json_string)
        else:
            # Handle case where no JSON is found
            return {"error": "Failed to get structured analysis from LLM"}

        # ==> STEP 4: Return the structured dictionary, ready for the frontend <==
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

    except (json.JSONDecodeError, AttributeError):
        # If parsing fails, return the raw text as a fallback
        return {
            "analysis": { "strengths": [raw_text_output] },
            "score": 0
        }