# from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import os
# import json

# # Load environment variables from a .env file
# load_dotenv()

# # Initialize the language model from OpenRouter
# # This setup uses the specified model and API credentials
# llm = ChatOpenAI(
#     model_name="deepseek/deepseek-r1-distill-llama-70b:free",
#     openai_api_key=os.getenv("OPENROUTER_API_KEY"),
#     openai_api_base="https://openrouter.ai/api/v1"
# )

# def generate_roadmap(parsed_data, analysis, job_description, current_score):
#     """
#     Generates a personalized career roadmap based on a candidate's resume,
#     a job description, and an analysis of how well they match.
#     """
    
#     # Define the prompt template that will be sent to the language model.
#     # It includes placeholders for the dynamic data.
#     prompt_template_string = """
# You are an expert career coach and learning advisor.

# Here is the candidate's parsed resume data:
# {parsed_data}

# Here is the resume analysis vs the job description (including strengths, gaps, and overall score):
# {analysis}

# Job Description:
# {job_description}

# The candidate currently has a matching score of {current_score}/100.

# Your task:
# ### 🎯 Goal
# Suggest **exactly 5–7 clear, actionable steps** to help the candidate reach a perfect score (100%).

# ### 📋 Format
# Return the roadmap in clean **Markdown** using this structure for **each point**:

# **1. [Step Title]**
# * **Why:** [Briefly explain why this step is important based on the analysis.]
# * **How:** [Provide concrete actions, resources, or projects.]
# * **Impact:** [Explain how this step will improve their score and close a specific gap.]
# * **Links:** [Based on the skill gap discussed above, search for **top-rated** and **up-to-date** learning resources that help improve this skill. Include:
# - At least one certification course (from Coursera, Udemy, edX, etc.)
# - One or two high-quality video tutorials (e.g., YouTube)
# - Optionally, free interactive platforms (like freeCodeCamp, W3Schools, etc.)

# For each resource, return:
# - Title
# - Short 1-line description
# - Direct URL

# Prefer beginner-to-intermediate level content. Avoid overly outdated links.]
# """
    
#     prompt = PromptTemplate.from_template(prompt_template_string)

#     # 1. Create the chain using the modern LCEL (LangChain Expression Language) syntax
#     # This pipes the output of the prompt directly into the language model.
#     chain = prompt | llm

#     # 2. Invoke the chain with a dictionary containing the required inputs
#     # The keys in the dictionary must match the variable names in the prompt template.
#     result_object = chain.invoke({
#         "parsed_data": json.dumps(parsed_data, indent=2),
#         "analysis": analysis,
#         "job_description": job_description,
#         "current_score": current_score
#     })

#     # 3. Return the .content attribute of the result to send back a clean string
#     return result_object.content

# # Example usage (optional, for testing):
# if __name__ == '__main__':
#     # You would replace these with actual data from your application
#     mock_parsed_data = {"experience": ["2 years as a junior developer"], "skills": ["Python", "Git"]}
#     mock_analysis = "Strengths: Python. Gaps: Lacks experience with cloud technologies like AWS and containerization (Docker) required by the job."
#     mock_job_description = "Seeking a mid-level developer with 3-5 years of experience, proficient in Python, AWS, and Docker."
#     mock_current_score = 65

#     # Generate the roadmap
#     roadmap = generate_roadmap(
#         parsed_data=mock_parsed_data,
#         analysis=mock_analysis,
#         job_description=mock_job_description,
#         current_score=mock_current_score
#     )
    
#     # Print the result
#     print("--- Generated Roadmap ---")
#     print(roadmap)




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
Create a 5-step roadmap. For each step, use this **Markdown format**:

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
