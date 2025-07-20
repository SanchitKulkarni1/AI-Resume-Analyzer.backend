from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    model_name="deepseek/deepseek-r1-distill-llama-70b:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

# Initialize the search tool
search_tool = TavilySearchResults(max_results=5)

def generate_roadmap(parsed_data, analysis, job_description, current_score):
    """
    Generates a personalized career roadmap by first finding real links
    and then using them to construct the final plan.
    """
    
    # --- Step 1: Use the LLM to identify the key skill gap ---
    # (For this example, we'll extract it from the analysis string, but an LLM could do this)
    skill_gap = "cloud technologies like AWS and containerization (Docker)"

    # --- Step 2: Use the Search Tool to find real, up-to-date links ---
    print(f"🔎 Searching for links related to: {skill_gap}...")
    search_query = f"top-rated courses and tutorials for {skill_gap} on Coursera, Udemy, and YouTube"
    found_links = search_tool.invoke(search_query) # This returns a list of strings
    formatted_links = "\n".join([f"- {link}" for link in found_links])
    
    # --- Step 3: Use the LLM with the retrieved links to generate the final roadmap ---
    final_prompt_template = """
You are an expert career coach. Your task is to create a 5-step roadmap using the candidate's data and the verified learning links provided below.

**Candidate Data:**
- Parsed Resume: {parsed_data}
- Resume Analysis: {analysis}
- Job Description: {job_description}
- Current Score: {current_score}/100

**Verified Learning Links (Use these in your response):**
{links}

### Instructions
Create a roadmap. For each step, use this **Markdown format**:

**[Step Number]. [Step Title]**
- **Why:** [Explain why this is important based on the analysis.]
- **How:** [Provide concrete actions and projects.]
- **Impact:** [Explain how this will improve their score.]
- **Links:**
  - [Title 1](https://...) – short description
  - [Title 2](https://...) – short description
"""
    
    final_prompt = PromptTemplate.from_template(final_prompt_template)
    
    chain = final_prompt | llm
    
    result_object = chain.invoke({
        "parsed_data": json.dumps(parsed_data, indent=2),
        "analysis": analysis,
        "job_description": job_description,
        "current_score": current_score,
        "links": formatted_links  # Pass the real links into the final prompt
    })
    
    return result_object.content
