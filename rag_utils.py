"""
Utilities for RAG operations including document retrieval, reranking, and formatting.
"""

import os
import json
import re
from typing import List, Dict, Optional, Tuple

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Debug printing function
def debug(message, debug_level=1):
    """Print debug message to stderr if debug level is enabled"""
    if debug_level > 0:
        import sys
        print(f"DEBUG: {message}", file=sys.stderr)

# LLM-based query analysis functions
def analyze_query_with_llm(query: str, last_exchange: Dict[str, str], api_key: str, task: str = "classify") -> str:
    """Use LLM to either classify query type or extract key terms"""
    if not query:
        return "new_topic" if task == "classify" else ""
        
    # Initialize LLM with lower temperature for more deterministic outputs
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        temperature=0.1,
        google_api_key=api_key
    )
    
    last_user_query = last_exchange.get("user", "") if last_exchange else ""
    last_assistant_response = last_exchange.get("assistant", "") if last_exchange else ""
    
    if task == "classify":
        prompt = f"""Analyze this conversation:
        
Previous user query: "{last_user_query}"
Previous assistant response (first 100 chars): "{last_assistant_response[:100]}..."
Current user query: "{query}"

Classify the current query as one of:
- "new_topic": An independent question not directly related to previous exchanges
- "followup": A question seeking more information about the previous topic
- "clarification": A question directly referring to something mentioned before

Return only one of these three classifications without explanation."""

    else:  # extract key terms
        prompt = f"""Extract 3-5 key terms from this text that would be helpful for retrieving relevant information:

"{query}"

Return only the key terms separated by spaces, without explanations or formatting."""

    try:
        response = llm.invoke(prompt)
        result = response.content.strip()
        
        # Handle classification task
        if task == "classify" and result not in ["new_topic", "followup", "clarification"]:
            # Default if not matching expected values
            return "new_topic"
        
        return result
    except Exception as e:
        debug(f"Error using LLM for query analysis: {e}")
        # Fallback results
        return "new_topic" if task == "classify" else ""

def classify_query_type(query: str, last_exchange: Dict[str, str], api_key: str) -> str:
    """Classify query as new topic, follow-up, or clarification using LLM"""
    return analyze_query_with_llm(query, last_exchange, api_key, task="classify")

def extract_key_terms(query: str, api_key: str) -> str:
    """Extract key terms from query using LLM"""
    return analyze_query_with_llm(query, {}, api_key, task="extract")

# Document retrieval functions
def get_vectorstore(api_key: str, chroma_path: str):
    """Initialize and return the vector store"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )
    
    return Chroma(
        persist_directory=chroma_path,
        collection_name="rag_collection",
        embedding_function=embeddings
    )

def retrieve_documents(
    query: str, 
    original_query: str, 
    query_type: str, 
    api_key: str,
    chroma_path: str,
    debug_level: int = 1
) -> List[Document]:
    """Retrieve documents based on query and query type"""
    debug(f"Retrieving documents for query: {query}", debug_level)
    debug(f"Query type: {query_type}", debug_level)
    
    # Load the vector store
    vectorstore = get_vectorstore(api_key, chroma_path)
    
    # Get initial documents based on query type
    docs = []
    
    if query_type == "new_topic":
        # For new topics, just use the current query
        docs = vectorstore.similarity_search(query, k=15)
        debug("Using standard retrieval for new topic", debug_level)
        
    elif query_type == "followup":
        # For follow-ups, try hybrid approach
        # 1. Retrieve based on enhanced query
        enhanced_docs = vectorstore.similarity_search(query, k=15)
        
        # 2. Also retrieve based on original query with higher k value
        original_docs = vectorstore.similarity_search(original_query, k=10)
        
        # Combine documents with deduplication (prioritize enhanced results)
        seen_ids = set()
        docs = []
        
        # First add enhanced results
        for doc in enhanced_docs:
            doc_id = doc.metadata.get("source", "") + str(doc.metadata.get("page_number", ""))
            if doc_id not in seen_ids:
                docs.append(doc)
                seen_ids.add(doc_id)
        
        # Then add original results if not already included
        for doc in original_docs:
            doc_id = doc.metadata.get("source", "") + str(doc.metadata.get("page_number", ""))
            if doc_id not in seen_ids and len(docs) < 10:
                docs.append(doc)
                seen_ids.add(doc_id)
                
        debug(f"Using hybrid retrieval for follow-up question, found {len(docs)} documents", debug_level)
        
    else:  # clarification
        # For clarifications, prioritize retrieval based on enhanced query
        docs = vectorstore.similarity_search(query, k=25)
        debug("Using enhanced retrieval for clarification", debug_level)
    
    debug(f"Retrieved {len(docs)} initial documents", debug_level)
    
    if not docs:
        return []
    
    if len(docs) <= 5:
        debug(f"Only {len(docs)} documents retrieved, skipping reranking", debug_level)
        return docs
    
    # Rerank documents with preference for tables if query seems to need data
    if any(term in query.lower() for term in ["data", "statistics", "numbers", "percentage", "figure", "table"]):
        docs = prioritize_tables(docs, debug_level)
    
    # Rerank remaining documents
    return rerank_documents(docs, query, api_key, debug_level)

def prioritize_tables(docs: List[Document], debug_level: int = 1) -> List[Document]:
    """Prioritize table chunks in the document list"""
    tables = [doc for doc in docs if doc.metadata.get("chunk_type") == "table"]
    non_tables = [doc for doc in docs if doc.metadata.get("chunk_type") != "table"]
    
    debug(f"Found {len(tables)} table chunks among {len(docs)} documents", debug_level)
    
    # Ensure we return a mix, but prioritize tables
    result = []
    result.extend(tables[:3])  # Up to 3 tables first
    result.extend(non_tables)  # Then add non-tables
    
    return result[:10]  # Return at most 10 documents

def rerank_documents(
    docs: List[Document], 
    query: str, 
    api_key: str,
    debug_level: int = 1
) -> List[Document]:
    """Rerank documents using Gemini LLM"""
    # Format documents for reranking with metadata
    formatted_docs = []
    for i, doc in enumerate(docs):
        meta = doc.metadata
        header = meta.get("header1", "")
        page = meta.get("page_number", "Unknown")
        chunk_type = meta.get("chunk_type", "text")
        
        # Include metadata in document representation
        doc_info = f"Document {i+1} [Page {page}"
        if header:
            doc_info += f", Section: {header}"
        if chunk_type == "table":
            doc_info += ", TABLE"
        doc_info += "]: "
        
        formatted_docs.append(f"{doc_info}{doc.page_content}")
    
    # Rerank with Gemini
    rerank_prompt = f"""
    You are a document reranking system specialized in financial reports. 
    Your task is to rerank the following documents based on their relevance to the query.
    Pay special attention to:
    1. Direct answers to the query
    2. Recent statistics and data points
    3. Tables that contain relevant information
    4. Section headers that match query terms
    
    Query: {query}
    
    Documents:
    {formatted_docs}
    
    Return a JSON list of document indices in descending order of relevance to the query.
    For example: [3, 1, 5, 2, 4] means document 3 is most relevant, followed by 1, etc.
    Only return the JSON list, no other text. Each index is ALWAYS an integer, not a string.
    """
    
    # Initialize reranker LLM
    reranker_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.1,
        google_api_key=api_key
    )
    
    debug("Sending reranking request to Gemini", debug_level)
    try:
        rerank_result = reranker_llm.invoke(rerank_prompt)
        debug(f"Reranking response: {rerank_result.content}", debug_level)
        
        # Extract indices from the response
        result_content = rerank_result.content.strip()
        start_idx = result_content.find("[")
        end_idx = result_content.rfind("]")
        
        if start_idx != -1 and end_idx != -1:
            indices_str = result_content[start_idx:end_idx+1]
            try:
                ordered_indices = json.loads(indices_str)
                debug(f"Parsed reranking indices: {ordered_indices}", debug_level)
                
                # Validate indices
                valid_indices = [idx-1 for idx in ordered_indices if 1 <= idx <= len(docs)]
                
                # Reorder documents
                reranked_docs = [docs[idx] for idx in valid_indices[:5]]
                debug(f"Returning {len(reranked_docs)} reranked documents", debug_level)
                return reranked_docs
            except Exception as e:
                debug(f"Error parsing reranking indices: {e}", debug_level)
        
        debug("Reranking failed, returning top-k original documents", debug_level)
        return docs[:5]
        
    except Exception as e:
        debug(f"Error during reranking: {e}", debug_level)
        return docs[:5]

# Query enhancement
def enhance_query_with_history(query: str, history: List[Dict[str, str]], api_key: str) -> str:
    """Enhance query with history based on query type detection"""
    if not history:
        return query
    
    # Get the last exchange
    last_exchange = history[-1] if history else None
    if not last_exchange:
        return query
    
    # Classify query type using LLM
    query_type = classify_query_type(query, last_exchange, api_key)
    
    # Apply enhancement strategy based on query type
    if query_type == "new_topic":
        # Independent question - use as is
        return query
        
    elif query_type == "followup":
        # Follow-up question - add key terms from last exchange
        key_terms = extract_key_terms(last_exchange.get("user", ""), api_key)
        # Combine but prioritize current query
        return f"{query} (context: {key_terms})"
        
    elif query_type == "clarification":
        # Multi-turn clarification - use more history
        # Include user's previous query and key points from assistant's response
        last_user_query = last_exchange.get("user", "")
        last_response = last_exchange.get("assistant", "")
        # Get first sentence of response to capture main point
        first_sentence = last_response.split(".")[0] if last_response else ""
        return f"{query} (referring to: {last_user_query} - {first_sentence})"
        
    # Default fallback
    return query

# Formatting functions
def format_sources(docs: List[Document]) -> str:
    """Format source documents with PDF-specific metadata"""
    sources = []
    seen_sources = set()  # To track unique sources
    
    for doc in docs:
        # Extract metadata
        metadata = doc.metadata
        title = metadata.get("title", "FHN Pulse 2024 Report")
        page = metadata.get("page_number", "Unknown")
        section = metadata.get("header1", "")
        chunk_type = metadata.get("chunk_type", "text")
        
        # Create source entry
        source_entry = f"* {title} (Page {page})"
        if section:
            source_entry += f", Section: {section}"
        if chunk_type == "table":
            source_entry += " [TABLE]"
        
        # Create a key for deduplication
        source_key = f"{title}-{page}-{section}"
        
        # Only add if we haven't seen this source before
        if source_key not in seen_sources:
            sources.append(source_entry)
            seen_sources.add(source_key)
    
    # Format as a string
    if sources:
        return "\n\nSources:\n" + "\n".join(sources)
    else:
        return "\n\nSources:\n* FHN Pulse 2024 Report (specific page unknown)"

def format_chat_history(history: List[Dict[str, str]], max_turns: int = 3) -> str:
    """Format the chat history for inclusion in the prompt"""
    if not history:
        return ""
    
    # Take only the most recent exchanges, limited by max_turns
    recent_history = history[-max_turns:]
    
    formatted_history = []
    for exchange in recent_history:
        user_message = exchange.get("user", "")
        ai_message = exchange.get("assistant", "")
        
        # Remove references section from AI messages for cleaner history
        ai_message = re.sub(r'\n+Sources:.*?$', '', ai_message, flags=re.DOTALL)
        
        formatted_history.append(f"User: {user_message}")
        formatted_history.append(f"Assistant: {ai_message}")
    
    return "\n".join(formatted_history)