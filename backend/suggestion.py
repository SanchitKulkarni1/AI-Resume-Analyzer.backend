from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
# No longer need to import LLMChain
import os

load_dotenv()

llm = ChatOpenAI(
    model_name="google/gemma-3-27b-it:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

def suggest_resume_improvements(resume_text):
    prompt = PromptTemplate.from_template("""
    You are a professional resume reviewer.
    
    Here is a candidate's resume content:
    ----
    {resume_text}
    ----
    
   Give **actionable, honest, and constructive feedback** to improve the resume in the following **6 categories**.

### 📋 Required Output Format (use Markdown syntax only):

## 1. 🧾 Formatting & Structure
- Comment on the overall layout, spacing, font usage, and alignment.
- Suggest structural improvements (e.g., consistent section headers, bullet spacing, etc.).

## 2. ✍️ Grammar & Clarity
- Point out unclear or awkward phrases.
- Recommend grammar or punctuation fixes.
- Suggest rewording for better readability.

## 3. 💥 Action Verbs & Metrics
- Suggest stronger verbs or quantifiable metrics where applicable.
- Highlight areas that lack impact.

## 4. 🧠 ATS Keyword Optimization
- Identify keywords missing based on the likely job target.
- Recommend where and how to add those.

## 5. 🧱 Missing Sections
- Suggest any essential resume sections that appear missing (e.g., Summary, Skills, Certifications, Projects).

## 6. 🎯 Tone & Professionalism
- Evaluate the overall tone for professionalism and confidence.
- Point out any overly casual or weak phrasing.

---

### 💡 Important Instructions:
- Use **headings** (`##`) and bullet points (`-`) for structure.
- Be **specific** about *what* to improve and *how* to do it.
- Keep the tone **professional but helpful**.
- **Do not** include any closing summary or extra comments.
- Return only the structured **Markdown content** as output.
""")

    # --- This is the new, updated section ---

    # 1. Create the chain using the pipe operator (LCEL)
    chain = prompt | llm

    # 2. Call the chain using .invoke() with a dictionary input
    # The key 'resume_text' must match the variable in your prompt template.
    result = chain.invoke({"resume_text": resume_text})
    return result.content 