"""
Title:
    ingest_data.py

Description:
    This script is responsible for ingesting the PDF documents, splitting them into chunks, 
    and adding them to the Chroma vector store.

Note:
    - Make sure to have the .env file with the OpenAI API key before running this script.
    - Run this script only once to avoid duplicate entries in the vector store.
"""

# ---------------------------------
# Importing necessary libraries
# ---------------------------------
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from dotenv import load_dotenv
load_dotenv()   # Load environment variables from .env file (OPENAI_API_KEY)

# ---------------------------------
# Configuration
# ---------------------------------
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"
EMBEDD_MODEL = r"text-embedding-3-large"
os.makedirs(CHROMA_PATH, exist_ok=True)

# ---------------------------------
# PDF Loading and Chunking
# ---------------------------------

# Loading the PDF document
loader = PyPDFDirectoryLoader(DATA_PATH)

# Load the documents into memory
raw_documents = loader.load()

# Splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=False,
)

# Creating the chunks
chunks = text_splitter.split_documents(raw_documents)

# Creating unique ID's
uuids = [str(uuid4()) for _ in range(len(chunks))]

# ---------------------------------
# Embeddings and Vector Store 
# --------------------------------- 

# Initiate the embeddings model
embeddings_model = OpenAIEmbeddings(model=EMBEDD_MODEL)

# Initiate the vector store
vector_store = Chroma(
    collection_name="attention_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Adding chunks to vector store
vector_store.add_documents(documents=chunks, ids=uuids)
