import os
import gradio as gr
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# import the .env file
from dotenv import load_dotenv
load_dotenv(dotenv_path="env/.env")

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Gemini's embedding model
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",   # Gemini embedding model
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# initiate the model
llm = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_AI_MODEL"), temperature=0.5)

# connect to the chromadb
vector_store = Chroma(
    collection_name="knowledge",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

# Set up the vectorstore to be the retriever
num_results = 3
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# call this function for every message added to the chatbot
def stream_response(message, history):
    #print(f"Input: {message}. History: {history}\n")

    # retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(message)

    # add all the chunks to 'knowledge'
    knowledge = ""

    for doc in docs:
        knowledge += doc.page_content+"\n\n"

    # make the call to the LLM (including prompt)
    if message is not None:

        partial_message = ""

        rag_prompt = f"""
        You are an assistant which answers questions based on knowledge which is provided to you.
        While answering, you don't use your internal knowledge, 
        but solely the information in the "The knowledge" section.
        You don't mention anything to the user about the povided knowledge.

        The question: {message}

        Conversation history: {history}

        The knowledge: {knowledge}

        """

        #print(rag_prompt)

        # stream the response to the Gradio App
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message

# initiate the Gradio app
chatbot = gr.ChatInterface(
    stream_response, 
    textbox=gr.Textbox(placeholder="✅ Send to the LLM...",
    container=False,
    autoscroll=True,
    scale=7),
)

# launch the Gradio app
chatbot.launch()