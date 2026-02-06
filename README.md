# 🧠 AI Resume Analyzer - Backend

A FastAPI-powered backend that analyzes resumes against job descriptions using AI/LLM models, providing structured parsing, compatibility scoring, improvement suggestions, and personalized upskilling roadmaps.

---

## 📁 Project Structure

```
backend/
├── main.py              # FastAPI application entry point & routes
├── app.py               # Legacy Flask implementation (deprecated)
├── extract_embed.py     # PDF extraction & Cohere embeddings
├── parsing_summary.py   # Resume parsing & job analysis via LLM
├── suggestion.py        # Resume improvement suggestions
├── roadmap.py           # Personalized upskilling roadmap generator
├── requirements.txt     # Python dependencies
├── render.yaml          # Render deployment configuration
├── .env                 # Environment variables (not committed)
├── chroma/              # ChromaDB vector store data
└── uploads/             # Temporary file uploads
```

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **Resume Parsing** | Extracts structured JSON data (name, skills, experience, education, etc.) from PDF resumes |
| **Job Match Analysis** | Analyzes resume vs job description with scoring (0-100) and detailed feedback |
| **Improvement Suggestions** | Provides actionable feedback on formatting, grammar, ATS optimization, and more |
| **Learning Roadmap** | Generates personalized upskilling paths with verified learning resources |

---

## 🔧 Core Modules

### `main.py`
The main FastAPI application with:
- **`GET /`** - Health check endpoint
- **`POST /analyze`** - Main resume analysis endpoint accepting:
  - `resume` (file) - PDF resume upload
  - `job_description` (form data) - Target job description

### `extract_embed.py`
Handles PDF processing and vector embeddings:
- `extract_text()` - Extracts text from PDF using pdfplumber
- `embed_resume()` - Creates Cohere embeddings stored in ChromaDB

### `parsing_summary.py`
LLM-powered resume analysis:
- `parse_resume()` - Extracts structured data (name, email, skills, education, etc.)
- `analyze_resume()` - RAG-based job match analysis with scoring rubric

### `suggestion.py`
Resume improvement recommendations across 6 categories:
1. Formatting & Structure
2. Grammar & Clarity
3. Action Verbs & Metrics
4. ATS Keyword Optimization
5. Missing Sections
6. Tone & Professionalism

### `roadmap.py`
Generates personalized learning roadmaps:
- Identifies skill gaps using LLM
- Searches for learning resources via Tavily API
- Creates step-by-step upskilling plans

---

## 🤖 AI Models Used

| Service | Model | Purpose |
|---------|-------|---------|
| OpenRouter | `mistralai/mistral-7b-instruct` | Resume parsing |
| OpenRouter | `deepseek/deepseek-r1-distill-llama-70b:free` | Roadmap generation |
| OpenRouter | `google/gemma-3-27b-it:free` | Improvement suggestions |
| Cohere | Embeddings API | Resume text embeddings |
| Tavily | Search API | Learning resource discovery |

---

## ⚙️ Environment Variables

Create a `.env` file in the `backend/` directory:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
COHERE_API_KEY=your_cohere_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

## 📦 Installation

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃 Running the Server

```bash
# Development mode
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

---

## 📡 API Reference

### `POST /analyze`

Analyzes a resume against a job description.

**Request:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "resume=@resume.pdf" \
  -F "job_description=Looking for a Python developer..."
```

**Response:**
```json
{
  "parsed": {
    "name": "John Doe",
    "email": "john@example.com",
    "skills": ["Python", "FastAPI", "Machine Learning"],
    "education": [...],
    "work_experience": [...],
    "projects": [...]
  },
  "analysis": {
    "strengths": [...],
    "improvements": [...],
    "matching_qualifications": "...",
    "missing_requirements": "...",
    "final_assessment": "..."
  },
  "score": 75,
  "suggestions": "## 1. 🧾 Formatting & Structure\n...",
  "roadmap": "**1. Learn Advanced Python**\n..."
}
```

---

## 🌐 Deployment

### Render

The project includes a `render.yaml` for easy deployment to Render:

```yaml
services:
  - type: web
    name: resume-analyzer-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## 🛠️ Tech Stack

- **Framework:** FastAPI
- **Vector Store:** ChromaDB
- **Embeddings:** Cohere
- **LLM Orchestration:** LangChain
- **PDF Processing:** pdfplumber
- **Search:** Tavily API

---
