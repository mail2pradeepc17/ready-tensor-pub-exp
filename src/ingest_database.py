import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

def main():
    # ---- Load environment variables ----
    DATA_PATH = r"./data/data.pdf"
    CHROMA_PATH = r"chroma_db"
    load_dotenv(dotenv_path="env/.env")

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("❌ GOOGLE_API_KEY not found. Please set it in your .env file.")

    # ---- Load PDFs ----
    loader = PyPDFLoader(DATA_PATH)  # just the folder path, no extra args
    docs = loader.load()

    # ---- Split into chunks ----
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000, 
        chunk_overlap=800, 
        separators=["\n\n", "\n", " ", ""])
    chunks = text_splitter.split_documents(docs)

    # ---- Gemini Embeddings ----
    embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=google_api_key)

    # ---- Create & persist Chroma DB ----
    persist_dir = CHROMA_PATH
    vectordb = Chroma.from_documents(   # from session state
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_dir)

    # Check vector dB contents
    print("✅ Total documents in Chroma:", vectordb._collection.count())
    print(f"✅ Vector database created at '{persist_dir}'")

if __name__ == "__main__":
    main()
