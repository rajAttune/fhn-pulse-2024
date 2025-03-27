"""
RAG Chatbot application entry point with Gradio UI.
"""

import os
import sys
import gradio as gr
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from graph import (
    State,
    initialize_state, 
    extract_query,
    create_retrieve_docs_node,
    create_generate_response_node
)
from rag_utils import debug

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set debug level (0 = no debug, 1 = basic debug info)
DEBUG_LEVEL = int(os.getenv("DEBUG_LEVEL", "1"))

# Get ChromaDB path from environment variables
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chromadb")

# Initialize the memory saver for persistence
memory = MemorySaver()

# Build the graph
def create_graph():
    """Create and compile the LangGraph"""
    graph_builder = StateGraph(State)
    
    # Create node functions with configuration
    retrieve_docs = create_retrieve_docs_node(
        api_key=GOOGLE_API_KEY,
        chroma_path=CHROMA_PATH,
        debug_level=DEBUG_LEVEL
    )
    
    generate_resp = create_generate_response_node(
        api_key=GOOGLE_API_KEY
    )
    
    # Add nodes
    graph_builder.add_node("initialize_state", initialize_state)
    graph_builder.add_node("extract_query", extract_query)
    graph_builder.add_node("retrieve_documents", retrieve_docs)
    graph_builder.add_node("generate_response", generate_resp)
    
    # Add edges
    graph_builder.add_edge(START, "initialize_state")
    graph_builder.add_edge("initialize_state", "extract_query")
    graph_builder.add_edge("extract_query", "retrieve_documents")
    graph_builder.add_edge("retrieve_documents", "generate_response")
    graph_builder.add_edge("generate_response", END)
    
    # Compile the graph with memory
    return graph_builder.compile(checkpointer=memory)

# Initialize the graph
graph = create_graph()

# Function to process user input and generate responses
def process_message(message, history):
    """Process a message through the graph and return the response"""
    debug(f"Received message: {message}", DEBUG_LEVEL)
    
    try:
        # Generate a unique thread ID based on the session
        thread_id = history[0][0] if history and history[0] else "new_session"
        config = {"configurable": {"thread_id": thread_id}}
        
        # Process the message through the graph
        debug("Processing message through graph", DEBUG_LEVEL)
        response = graph.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config
        )
        debug("Response generated successfully", DEBUG_LEVEL)
        
        # Extract the AI response
        ai_response = response["messages"][-1].content
        return ai_response
    except Exception as e:
        error_msg = f"Error processing your question: {str(e)}"
        debug(f"ERROR: {error_msg}", DEBUG_LEVEL)
        return error_msg

# Create the Gradio interface
def create_interface():
    """Create and return the Gradio chat interface"""
    debug("Setting up Gradio ChatInterface", DEBUG_LEVEL)
    bot = gr.ChatInterface(
        fn=process_message,
        title="FHN Pulse 2024 Report Chatbot",
        description="Ask questions about the Financial Health Network Pulse 2024 report on financial health in the United States.",
        examples=[
            "What are the key findings of the FHN Pulse 2024 report?",
            "How did financial health trends change in 2024?",
            "What recommendations does the report make for financial institutions?"
        ]
    )
    return bot

# Launch the Gradio interface
if __name__ == "__main__":
    debug(f"Using ChromaDB from: {CHROMA_PATH}", DEBUG_LEVEL)
    debug("Launching Gradio interface", DEBUG_LEVEL)
    bot = create_interface()
    bot.launch(share=True)