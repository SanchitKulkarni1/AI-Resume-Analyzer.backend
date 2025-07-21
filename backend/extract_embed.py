# import uuid
# import pdfplumber
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from dotenv import load_dotenv
# from fastapi import UploadFile
# import os

# # Load env vars
# load_dotenv()

# # Define persistent path for Render (Render supports /mnt/data for disks)
# CHROMA_PATH = "/mnt/data/chroma"  # You must mount this in your Render settings

# # ✅ Extract text from PDF
# async def extract_text(file: UploadFile) -> str:
#     try:
#         file.file.seek(0)
#         with pdfplumber.open(file.file) as pdf:
#             return "\n".join(page.extract_text() or "" for page in pdf.pages)
#     except Exception as e:
#         raise ValueError(f"Failed to parse PDF: {e}")

# # ✅ Embed and store resume text
# async def embed_resume(resume_text: str):
#     embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#     vectordb = Chroma.from_texts(
#         [resume_text],
#         embedding=embedding_model,
#     )

#     return vectordb


import uuid
import pdfplumber
import os
from fastapi import UploadFile
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import CohereEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Chroma DB path (recommended for Render Free Tier) # or just "chroma" locally

# Load the Cohere embedding model once
embedding_model = CohereEmbeddings(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    user_agent="my-resume-analyzer/1.0"
)

# ✅ PDF Text Extraction
async def extract_text(file: UploadFile) -> str:
    try:
        file.file.seek(0)
        with pdfplumber.open(file.file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {e}")

# ✅ Embedding Function
async def embed_resume(resume_text: str):
    try:
        vectordb = Chroma.from_texts(
            [resume_text],
            embedding=embedding_model,
        )
        return vectordb
    except Exception as e:
        raise RuntimeError(f"Failed to create embedding: {e}")
