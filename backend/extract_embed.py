import pdfplumber
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

#extract resume
def extract_text(filepath):
    with pdfplumber.open(filepath) as pdf:
        return "\n".join (page.extract_text() or "" for page in pdf.pages)
    
#embed and store resume text
def embed_resume(resume_text):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts (
        [resume_text],
        embedding=embedding_model,
        persist_directory="./chroma"
    )

    return vectordb