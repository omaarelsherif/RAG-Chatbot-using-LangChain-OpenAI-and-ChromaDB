"""
Title:
    chatbot.py

Description:
    This script implements a Retrieval-Augmented Generation (RAG) chatbot
    using LangChain, OpenAI models, Chroma vector database, and Gradio UI.

Note:
    Make sure to have the .env file with the OpenAI API key before running this script.
"""

# ---------------------------------
# Importing necessary libraries
# ---------------------------------
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr
from dotenv import load_dotenv
load_dotenv()   # Load environment variables from .env file (OPENAI_API_KEY)

# ---------------------------------
# Configuration
# ---------------------------------
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# ---------------------------------
# Embeddings and LLM model 
# --------------------------------- 

# Used to convert text into vector embeddings for similarity search
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Initiate the LLM model
llm = ChatOpenAI(temperature=0.5, model='gpt-4o-mini')

# ---------------------------------
# Vector store 
# --------------------------------- 

# Connect to the chromadb
vector_store = Chroma(
    collection_name="attention_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

# Set up the vectorstore to be the retriever
num_results = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# Call this function for every message added to the chatbot
def stream_response(message, history):
    """"""

    # Retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(message)

    # Add all the chunks to 'knowledge'
    knowledge = ""

    for doc in docs:
        knowledge += doc.page_content+"\n\n"

    # Make the call to the LLM (including prompt)
    if message is not None:

        partial_message = ""

        rag_prompt = f"""
        You are a strict RAG assistant.

        Rules:
        - Use ONLY the provided context.
        - If context is insufficient, say "I don't know based on the document."
        - Do not use external knowledge.

        Context:
        {knowledge}

        Question:
        {message}

        Answer:
        """

        # Stream the response to the Gradio App
        for response in llm.stream(rag_prompt):
            partial_message += response.content # type: ignore
            yield partial_message

# Initiate the Gradio app
chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
    container=False,
    autoscroll=True,
    scale=7),
)

# Launch the chatbot Gradio app
chatbot.launch()
