import json
import re
import logging
from dotenv import load_dotenv
from typing import Any, Dict

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    status,
)
from fastapi.middleware.cors import CORSMiddleware

from extract_embed import extract_text
from parsing_summary import parse_resume, analyze_resume
from suggestion import suggest_resume_improvements
from roadmap import generate_roadmap

# --------------------------------------------------
# Setup
# --------------------------------------------------

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("resume-analyzer")

app = FastAPI(title="Resume Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Robust JSON extractor for LLM output.
    Supports:
    - clean JSON
    - JSON wrapped in text
    - markdown-fenced JSON
    """

    # Remove markdown fences
    text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try object
    obj_match = re.search(r"\{[\s\S]*?\}", text)
    if obj_match:
        return json.loads(obj_match.group())

    raise ValueError("No valid JSON found in LLM output")


async def parse_resume_with_retry(resume_text: str, retries: int = 1) -> Dict[str, Any]:
    """
    Calls parse_resume LLM and enforces JSON contract with retry.
    """
    last_error = None

    for attempt in range(retries + 1):
        raw_output = await parse_resume(resume_text)

        try:
            return extract_json_from_text(raw_output)
        except Exception as e:
            logger.error("❌ Resume parsing failed (attempt %d)", attempt + 1)
            logger.error("RAW LLM OUTPUT:\n%s", raw_output)
            last_error = e

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to parse resume into structured JSON."
    )


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
async def health_check():
    return {"status": "Resume Analyzer API running"}


@app.post("/analyze")
async def analyze_resume_endpoint(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    logger.info("📥 /analyze request received")

    if not resume.filename:
        raise HTTPException(400, "Resume file missing")

    try:
        # 1️⃣ Extract resume text
        resume_text = await extract_text(resume)
        logger.info("📄 Extracted text from %s", resume.filename)

        # 2️⃣ Parse resume → structured JSON (with retry)
        logger.info("🧠 Parsing resume with LLM")
        parsed_resume = await parse_resume_with_retry(resume_text, retries=1)

        # 3️⃣ Analyze vs job description
        logger.info("📊 Analyzing resume vs JD")
        analysis = await analyze_resume(resume_text, job_description)

        score = analysis.get("score")
        analysis_text = analysis.get("analysis")

        # 4️⃣ Suggestions
        logger.info("💡 Generating improvement suggestions")
        suggestions = await suggest_resume_improvements(resume_text)

        # 5️⃣ Roadmap
        logger.info("🛣️ Generating learning roadmap")
        roadmap = await generate_roadmap(
            parsed_resume,
            analysis_text,
            job_description,
            score
        )

        logger.info("✅ Analysis completed successfully")

        return {
            "parsed": parsed_resume,
            "analysis": analysis_text,
            "score": score,
            "suggestions": suggestions,
            "roadmap": roadmap,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("🔥 Unexpected error in /analyze")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while analyzing resume."
        )
