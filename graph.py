"""
LangGraph state definition and node functions for RAG chatbot.
"""

import re
from typing import Dict, List, Optional, Annotated, TypedDict

from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.graph.message import add_messages

from rag_utils import (
    debug,
    enhance_query_with_history,
    classify_query_type,
    retrieve_documents,
    format_sources,
    format_chat_history
)

# Define the state for our LangGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]
    query: Optional[str]
    original_query: Optional[str]  # Store original query before enhancement
    retrieval_documents: Optional[List[Document]]
    results: Optional[str]
    history: List[Dict[str, str]]  # Store conversation history
    query_type: Optional[str]  # Store query classification (new, followup, clarification)

# Node functions for the graph
def initialize_state(state: State) -> Dict:
    """Initialize state with empty history if not present"""
    if "history" not in state:
        return {"history": []}
    return {}

def extract_query(state: State) -> Dict:
    """Extract the query from the latest message and enhance with history if needed"""
    messages = state["messages"]
    history = state.get("history", [])
    
    if not messages:
        return {"query": None}
    
    # Get the most recent user message
    current_query = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or msg.get("role") == "user":
            current_query = msg.content if hasattr(msg, "content") else msg.get("content", "")
            debug(f"Extracted current query: {current_query}")
            break
    
    if not current_query:
        return {"query": None}
        
    # Handle different query enhancement strategies based on query type
    enhanced_query = enhance_query_with_history(current_query, history)
    debug(f"Enhanced query: {enhanced_query}")
    
    return {
        "query": enhanced_query,
        "original_query": current_query  # Store original for prompt
    }

def retrieve_docs_node(state: State, api_key: str, chroma_path: str, debug_level: int) -> Dict:
    """Node for retrieving documents from the vector store"""
    query = state.get("query")
    original_query = state.get("original_query", query)
    
    if not query:
        debug("No query to retrieve documents for")
        return {"retrieval_documents": []}
        
    # Classify query type
    history = state.get("history", [])
    last_exchange = history[-1] if history else None
    query_type = "new_topic"
    
    if last_exchange:
        query_type = classify_query_type(original_query, last_exchange)
        
    debug(f"Query type classified as: {query_type}")
    
    # Retrieve documents
    docs = retrieve_documents(
        query=query,
        original_query=original_query,
        query_type=query_type,
        api_key=api_key,
        chroma_path=chroma_path,
        debug_level=debug_level
    )
    
    return {
        "retrieval_documents": docs,
        "query_type": query_type
    }

def generate_response(state: State, api_key: str) -> Dict:
    """Generate a response using the retrieved documents as context and chat history"""
    messages = state.get("messages", [])
    docs = state.get("retrieval_documents", [])
    query = state.get("original_query", state.get("query", ""))  # Use original query for response
    history = state.get("history", [])
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        google_api_key=api_key
    )
    
    if not query or not docs:
        debug("No query or documents to generate response from")
        # Just use the raw LLM if no context
        ai_response = llm.invoke(messages)
        
        # Update history
        new_exchange = {"user": query, "assistant": ai_response.content}
        updated_history = history + [new_exchange]
        
        return {
            "messages": [ai_response],
            "history": updated_history
        }
    
    # Create a context string from the documents
    context_str = "\n\n".join([doc.page_content for doc in docs])
    
    # Format sources in the required format
    sources_str = format_sources(docs)
    
    # Format chat history (last 3 exchanges)
    history_str = format_chat_history(history, max_turns=3)
    
    # Get query type for prompt customization
    query_type = state.get("query_type", "new_topic")
    
    # Custom prompt template - modified to include PDF structure awareness
    template = """You are a financial health expert and consultant designed to
    provide insights from the FHN Pulse 2024 report on financial health in the United States.

    You respond based ONLY on what you find in your knowledge base, which is
    the Financial Health Network (FHN) Pulse 2024 report on the state of financial health in
    the United States. 
    
    If you don't find something in the knowledge base,
    just say so, and don't make up anything else!!!
    
    When referencing information, include page numbers when available.
    Pay special attention to tables and figures in the report.
    
    Summarize the documents you find and respond 
    balancing a conversational and professional tone. 
    
    Bias to provide more information rather than less, anticipating follow-up questions.

    
    {history_section}
    
    Context information is below.
    ---------------------
    {context}
    ---------------------

    Given the context information and not prior knowledge, answer the question: {question}
    
    Query type: {query_type}
    """
    
    # Add history section if history exists
    history_section = ""
    if history_str:
        history_section = f"""Previous conversation:
    ---------------------
    {history_str}
    ---------------------"""
    
    prompt = PromptTemplate(
        template=template, 
        input_variables=["context", "question", "history_section", "query_type"]
    )
    
    formatted_prompt = prompt.format(
        context=context_str, 
        question=query, 
        history_section=history_section,
        query_type=query_type
    )
    debug("Sending prompt to Gemini")
    
    # Get response from LLM
    response = llm.invoke(formatted_prompt)
    content = response.content
    
    # Remove any references section that might have been generated despite instructions
    content = re.sub(r'\n+References:.*?$', '', content, flags=re.DOTALL)
    
    # Append our properly formatted sources
    final_content = content + sources_str
    
    # Create AIMessage
    ai_message = AIMessage(content=final_content)
    
    # Update history with the new exchange
    new_exchange = {"user": query, "assistant": final_content}
    updated_history = history + [new_exchange]
    
    return {
        "messages": [ai_message],
        "history": updated_history
    }

# Wrapper function to create the retrieve_docs node with configuration
def create_retrieve_docs_node(api_key: str, chroma_path: str, debug_level: int):
    """Create a retrieve_docs node with configuration injected"""
    def _retrieve_docs_wrapped(state: State) -> Dict:
        return retrieve_docs_node(state, api_key, chroma_path, debug_level)
    return _retrieve_docs_wrapped

# Wrapper function to create the generate_response node with configuration
def create_generate_response_node(api_key: str):
    """Create a generate_response node with API key injected"""
    def _generate_response_wrapped(state: State) -> Dict:
        return generate_response(state, api_key)
    return _generate_response_wrapped