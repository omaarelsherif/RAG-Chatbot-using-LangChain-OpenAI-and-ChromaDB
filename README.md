# RAG Chatbot using LangChain, OpenAI, and ChromaDB

This project is a **Retrieval-Augmented Generation (RAG) chatbot** that allows users to ask questions about PDF documents (e.g., research papers) and get intelligent answers based only on the document content.

It uses **LangChain, OpenAI, and ChromaDB** to combine document retrieval with large language models.

---

## System Architecture (RAG Pipeline)

This diagram illustrates how documents are ingested, embedded, stored, and later retrieved to generate context-aware responses using a Retrieval-Augmented Generation (RAG) pipeline.
<img src="images/Project Architecture.png"/>

---

## How It Works

1. PDF file are loaded from the `data/` folder
2. Documents are split into overlapping chunks
3. Each chunk is converted into embeddings
4. Embeddings are stored in a Chroma vector database locally
5. When a user asks a question:

   * The system retrieves the most relevant chunks
   * These chunks are passed to the LLM
   * The model generates an answer based ONLY on retrieved context

---

## Project Structure

```
.
├── ingest_data.py     # Loads PDFs and builds vector database
├── app.py             # RAG chatbot with Gradio app
├── data/              # PDF document
├── chroma_db/         # Vector database storage

```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/omaarelsherif/RAG-Chatbot-using-LangChain-OpenAI-and-ChromaDB.git
```

### 2. Initialize virtual environment using UV

```bash
uv init
```
```bash
uv sync
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 4. Add OpenAI API key

Create a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

---

## Ingest Documents

Run this script once to process the PDF and build the vector database:

```bash
python ingest_data.py
```

---

## Run the Chatbot

Start the Gradio UI:

```bash
python app.py
```

Then open the local link in your browser.

---

## Example Use Cases

* Ask questions about research papers
* Summarize sections of PDFs
* Extract key ideas from academic documents
* Build document-aware AI assistants

---

## Tech Stack

* Python
* LangChain
* OpenAI GPT-4o-mini
* ChromaDB
* Gradio
* PyPDF
