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
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from fastapi import UploadFile
from langchain.embeddings import HuggingFaceEmbeddings
import os

# Load env vars
load_dotenv()

# ✅ CHROMA storage path (for Render: use /mnt/data)
  # Make sure Render uses a mounted disk for this path

# ✅ Extract text from PDF
async def extract_text(file: UploadFile) -> str:
    try:
        file.file.seek(0)
        with pdfplumber.open(file.file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {e}")

# ✅ Embed and store resume text using HuggingFace
async def embed_resume(resume_text: str):
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectordb = Chroma.from_texts(
            [resume_text],
            embedding=embedding_model,
        )

        return vectordb
    except Exception as e:
        raise RuntimeError(f"Failed to create embedding: {e}")
