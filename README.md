# PDF-RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot system for querying information from PDF documents using LangGraph and Google's Gemini models.

## Overview

This system consists of two main components:

1. **PDF Processing Pipeline** (`pdf_to_rag.py`): Extracts text and images from PDF documents, processes them, and stores in a ChromaDB vector database.
2. **RAG Chatbot** (`app.py`, `graph.py`, `rag_utils.py`): A LangGraph-powered chat interface that retrieves relevant information from the vector database to answer user questions about the PDF content.

## How to Run

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages include:
- langchain, langchain-google-genai, langchain-chroma
- langgraph
- pymupdf, pymupdf4llm
- chromadb
- gradio
- dotenv
- fitz

### Step 2: Set Up Environment Variables

Create a `.env` file in the project root with your API keys:

```
GOOGLE_API_KEY=your_gemini_api_key_here
# Optional: ANTHROPIC_API_KEY=your_claude_api_key_here
```

### Step 3: Process PDFs and Create Vector Database

```bash
python pdf_to_rag.py --input ./sources --output ./chromadb --debug 1
```

Parameters:
- `--input`: Directory containing PDF files to process (default: `./sources`)
- `--output`: Output directory for ChromaDB (default: `./chromadb`)
- `--debug`: Debug level (0=none, 1=basic, 2=verbose)
- `--force`: Force recreation of outputs even if they already exist

### Step 4: Run the Chatbot Interface

```bash
python app.py
```

Parameters:
- `--knowledge`: Optional path to additional knowledge base text file
- `--chroma`: Path to ChromaDB directory (default: `./chromadb`)
- `--debug`: Debug level (0=none, 1=basic, 2=verbose)

## How It Works

### PDF to Vector Database Pipeline

The PDF processing pipeline (`pdf_to_rag.py`) implements a comprehensive workflow:

1. **Text Extraction**:
   - Uses PyMuPDF and PyMuPDF4LLM to extract text while preserving document structure
   - Converts PDF content to markdown format with headers, lists, and tables
   - Detects document structure (title, sections, etc.)

2. **Semantic Chunking**:
   - First splits text based on markdown headers
   - Further splits content using a recursive character splitter
   - Special handling for tables (keeps them intact)
   - Enriches chunks with metadata (source, page number, section headers)

3. **Image Processing**:
   - Extracts images from PDF documents
   - Filters out small or decorative images
   - Uses Google's Gemini model to generate descriptions and extract insights
   - Creates separate vector embeddings for image content

4. **Vector Database Creation**:
   - Embeds chunks using Google's text-embedding-004 model
   - Stores embeddings and metadata in ChromaDB
   - Handles deduplication of content

### RAG Chatbot with LangGraph

The chatbot component implements a LangGraph-based query pipeline:

1. **Graph Structure** (`app.py`):
   - Creates a `StateGraph` with nodes for initialization, query extraction, document retrieval, and response generation
   - Handles conversation state persistence using `MemorySaver`
   - Provides a Gradio chat interface

2. **State Management** (`graph.py`):
   - Defines a `State` TypedDict with messages, query, documents, and history
   - Manages conversation context across multiple turns
   - Tracks query types (new topic, followup, clarification)

3. **Advanced Retrieval** (`rag_utils.py`):
   - **Query Understanding**:
     - Uses LLM to classify query types (new topic, followup, clarification)
     - Enhances queries by incorporating conversation history
   
   - **Context Retrieval**:
     - Retrieves relevant documents based on query type
     - Uses different retrieval strategies for new topics vs. followups
     - Prioritizes tables for data-oriented questions
   
   - **Reranking**:
     - Uses Gemini to rerank retrieved documents
     - Considers document types, page numbers, and section relevance
     - Ensures diverse and relevant information

4. **Response Generation**:
   - Builds detailed prompt with retrieved context and conversation history
   - Generates response using Gemini-2.0-flash
   - Formats response with source references

## Query Retrieval and Response Strategies

The system employs several advanced strategies:

### Query Analysis

- **Query Classification**: Uses Gemini to classify queries as new topics, followups, or clarifications
- **Query Enhancement**: Incorporates conversation context for better retrieval, especially for followup questions
- **Key Term Extraction**: Extracts important terms to improve retrieval quality

### Retrieval Optimization

- **Adaptive Retrieval**: Uses different strategies based on query type:
  - New topics: Standard similarity search
  - Followups: Hybrid retrieval combining enhanced and original queries
  - Clarifications: Expanded context search with higher k value

- **Content Prioritization**:
  - Detects data-oriented queries and prioritizes table chunks
  - Balances text and image content for comprehensive responses

- **LLM-based Reranking**:
  - Uses Gemini to rerank retrieved documents based on relevance
  - Considers document metadata, content type, and query terms
  - Limits to top-k most relevant documents for focused responses

### Response Generation

- **Contextual Prompting**:
  - Builds detailed prompts with retrieved documents, conversation history, and query type
  - Includes specific guidance for handling different query types
  - References source documents with page numbers and sections

- **Source Attribution**:
  - Includes numbered references in responses
  - Provides document sources with page numbers
  - Deduplicates sources for cleaner presentation

## Customization

- The system can be extended with additional knowledge by using the `--knowledge` parameter
- Models can be switched (code includes commented sections for using Claude instead of Gemini)
- Debug levels allow for different verbosity of logging
- The Gradio interface can be customized with examples and styling

## Notes

- The system is designed for financial reports but can be adapted for any domain
- Performance depends on the quality of the PDFs and their content structure
- The system handles multi-turn conversations and maintains context across turns
- Image processing requires significant computational resources, especially for large PDFs
