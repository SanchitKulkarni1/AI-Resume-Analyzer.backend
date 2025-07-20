# main.py - Your new FastAPI application

import os
import json
import re
import logging
from dotenv import load_dotenv

from fastapi import (
    FastAPI, 
    UploadFile, 
    File, 
    Form, 
    HTTPException, 
    status
)
from fastapi.middleware.cors import CORSMiddleware

# --- IMPORTANT ---
# You must update these imported functions to be 'async def' functions
# and use an async library like 'httpx' for API calls inside them.
from extract_embed import extract_text
from parsing_summary import analyze_resume, parse_resume
from suggestion import suggest_resume_improvements
from roadmap import generate_roadmap

# Load environment variables from a .env file
load_dotenv()

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the FastAPI app
app = FastAPI()

# --- CORS Middleware ---
# This allows your frontend to communicate with this backend.
# Update 'allow_origins' to your specific frontend URL in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for now, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Resume Analyzer API is running!"}


@app.post("/analyze")
async def analyze_resume_endpoint(
    # FastAPI uses dependency injection to get the file and form data.
    # This is more secure and provides better validation than Flask's request object.
    resume: UploadFile = File(..., description="The user's resume file (PDF, DOCX)."), 
    job_description: str = Form(..., description="The job description to compare against.")
):
    """
    Analyzes a resume against a job description, providing a score,
    analysis, suggestions, and a roadmap for improvement.
    """
    logger.info("Received request for /analyze")

    # FastAPI handles basic validation, but we can add checks.
    if not resume or not resume.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="No resume file uploaded."
        )
    
    if not job_description:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Job description is required."
        )

    try:
        # Read the file content asynchronously
        file_contents = await resume.read()
        
        # This function should be able to handle bytes from the uploaded file
        resume_text = await extract_text(resume)
        logger.info(f"Extracted text from {resume.filename}")

        # --- Asynchronous LLM Calls ---
        # Each of these functions should now be an 'async def' function.
        # 'await' pauses this function, allowing the server to handle other requests.
        logger.info("Calling LLM for parsing...")
        raw_parsed_string = await parse_resume(resume_text)
        
        logger.info("Calling LLM for analysis...")
        analysis_results_dict = await analyze_resume(resume_text, job_description)
        
        logger.info("Calling LLM for suggestions...")
        suggestions_string = await suggest_resume_improvements(resume_text)

        score = analysis_results_dict.get("score")
        analysis_text = analysis_results_dict.get("analysis")
        
        # Find the JSON object within the potentially messy LLM string
        match = re.search(r"\{.*\}", raw_parsed_string, re.DOTALL)
        if not match:
            logger.error("LLM output for parsing did not contain a valid JSON object.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="Failed to parse resume content from LLM output."
            )
        
        clean_json_string = match.group(0)
        parsed_data = json.loads(clean_json_string)

        logger.info("Calling LLM for roadmap generation...")
        roadmap_string = await generate_roadmap(parsed_data, analysis_text, job_description, score)

        # Assemble the final response
        final_response = {
            "parsed": parsed_data,
            "analysis": analysis_text,
            "score": score,
            "suggestions": suggestions_string,
            "roadmap": roadmap_string
        }
        
        logger.info("Successfully completed analysis.")
        return final_response

    except json.JSONDecodeError:
        logger.error("LLM returned malformed JSON after cleaning.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="LLM returned malformed JSON."
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred in /analyze: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"An unexpected error occurred: {e}"
        )

# Note: You don't need the 'if __name__ == "__main__":' block.
# You will run this server using Gunicorn in production.