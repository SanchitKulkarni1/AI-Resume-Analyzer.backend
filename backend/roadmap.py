from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    model_name="deepseek/deepseek-r1-distill-llama-70b:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

search_tool = TavilySearchResults(max_results=5)

# STEP 1: Extract Skill Gaps
extract_gap_prompt = PromptTemplate.from_template("""
You are an expert career advisor.

Based on the resume analysis and job description below, identify up to 3 specific technical or professional skill gaps that prevent the candidate from achieving a perfect match score.

**Resume Analysis:**
{analysis}

**Job Description:**
{job_description}

Return your answer as a plain comma-separated string of skill gaps.
""")

# STEP 2: Generate the Roadmap Prompt (Dynamic Steps)
roadmap_prompt = PromptTemplate.from_template("""
You are an expert career mentor.

Create a detailed, personalized upskilling roadmap to help this candidate go from their **current score of {current_score}/100 to a full 100** in job-readiness. Use the resume data, analysis, job description, and verified learning links below. You may use **as many steps as needed** – don’t constrain to 5 if more are needed for real growth.

**Candidate Data:**
- Parsed Resume: {parsed_data}
- Resume Analysis: {analysis}
- Job Description: {job_description}
- Current Score: {current_score}

**Skill Gaps:**
{skill_gaps}

**Verified Learning Links:**
{links}

### Instructions
For each step, use this markdown format:

**[Step Number]. [Step Title]**
- **Why:** Explain why this step matters.
- **How:** Give concrete actions like projects, courses, or practices.
- **Impact:** Explain how this improves their job readiness score.
- **Links:**
{{
  if relevant, list learning resources with format:
  - [Course Title](https://link.com) – one-line summary
}}

Only include high-impact, personalized steps. The roadmap must be realistic, motivating, and highly actionable.
""")

async def generate_roadmap(parsed_data, analysis, job_description, current_score):
    # Step 1: Extract relevant skill gaps
    gap_chain = extract_gap_prompt | llm | StrOutputParser()
    skill_gaps_text = gap_chain.invoke({
        "analysis": analysis,
        "job_description": job_description
    })

    print(f"🔍 Skill gaps found: {skill_gaps_text}")

    # Step 2: Use Tavily to find learning resources for each gap
    all_links = []
    for gap in skill_gaps_text.split(","):
        search_query = f"best online courses, tutorials, or projects to learn {gap.strip()}"
        print(f"🌐 Searching resources for: {gap.strip()}")
        results = search_tool.invoke(search_query)
        all_links.extend(results)

    formatted_links = "\n".join([f"- {link}" for link in all_links])

    # Step 3: Generate dynamic, high-quality roadmap
    roadmap_chain = roadmap_prompt | llm

    result = roadmap_chain.invoke({
        "parsed_data": json.dumps(parsed_data, indent=2),
        "analysis": analysis,
        "job_description": job_description,
        "current_score": current_score,
        "skill_gaps": skill_gaps_text,
        "links": formatted_links
    })

    return result.content
