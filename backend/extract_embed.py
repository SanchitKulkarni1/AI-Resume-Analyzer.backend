import pdfplumber
import io
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from fastapi import UploadFile

load_dotenv()

#extract resume
async def extract_text(file: UploadFile) -> str:
    try:
        # Don’t read as bytes, just pass the file-like object directly
        file.file.seek(0)  # ensure start of file
        with pdfplumber.open(file.file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {e}")
    
#embed and store resume text
async def embed_resume(resume_text):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts (
        [resume_text],
        embedding=embedding_model,
        persist_directory="./chroma"
    )

    return vectordb