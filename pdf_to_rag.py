#!/usr/bin/env python3
"""
PDF to Vector Database Converter for RAG

This script extracts text from multiple PDF files using PyMuPDF4LLM, preserves document structure,
creates semantic chunks, and stores them in a ChromaDB vector database for RAG.

Usage:
    python pdf_to_rag.py [--input SOURCES_DIR] [--output CHROMADB_DIR] [--debug LEVEL] [--force]
"""

import os
import sys
import argparse
import re
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
from dotenv import load_dotenv

# PyMuPDF4LLM for PDF extraction
try:
    import pymupdf4llm
except ImportError:
    print("Error: PyMuPDF4LLM is required. Install with: pip install pymupdf4llm", file=sys.stderr)
    sys.exit(1)

# ChromaDB and embedding dependencies
try:
    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
    from langchain.docstore.document import Document
    import chromadb
    from dotenv import load_dotenv
except ImportError:
    print("Error: Required dependencies missing. Install with:", file=sys.stderr)
    print("pip install langchain langchain-chroma langchain-google-genai chromadb python-dotenv", file=sys.stderr)
    sys.exit(1)

# Debug function
def debug(message, level, current_level):
    """Print debug message to stderr if current_level meets or exceeds required level"""
    if current_level >= level:
        print(f"DEBUG [{level}]: {message}", file=sys.stderr)

class PDFProcessor:
    def __init__(self, pdf_path: str, debug_level: int = 1, force: bool = False):
        """Initialize the PDF processor for a single file"""
        self.pdf_path = pdf_path
        self.debug_level = debug_level
        self.force = force
        
        # Setup output paths
        self.pdf_dir = os.path.dirname(os.path.abspath(pdf_path))
        self.pdf_name = os.path.basename(pdf_path)
        self.base_name = os.path.splitext(self.pdf_name)[0]
        self.md_path = os.path.join(self.pdf_dir, f"{self.base_name}.md")
        self.chunks_path = os.path.join(self.pdf_dir, f"{self.base_name}_chunks.txt")
        
        # Extract document information
        self.title = self.base_name.replace("_", " ").title()
        self.org_name = "Unknown Organization"  # This would need to be set manually or extracted
        
        # Document content
        self.markdown_content = ""
        self.chunks = []
        
        # Document for page count
        try:
            import pymupdf
            self.doc = pymupdf.open(self.pdf_path)
        except Exception as e:
            debug(f"Error opening PDF with PyMuPDF: {str(e)}", 0, self.debug_level)
            self.doc = None

    def extract_text(self) -> str:
        """Extract text from PDF and convert to markdown"""
        debug(f"Extracting text from {self.pdf_path}", 1, self.debug_level)
        
        # Check if markdown already exists and not forcing recreation
        if os.path.exists(self.md_path) and not self.force:
            debug(f"Markdown file already exists at {self.md_path}. Loading...", 1, self.debug_level)
            with open(self.md_path, 'r', encoding='utf-8') as f:
                self.markdown_content = f.read()
            return self.markdown_content
        
        # Extract markdown from PDF
        debug("Extracting text and converting to markdown", 1, self.debug_level)
        try:
            # Use the to_markdown function from pymupdf4llm
            self.markdown_content = pymupdf4llm.to_markdown(
                self.pdf_path,
                table_strategy="lines",  # More flexible table detection
                write_images=False,      # Don't extract images yet
                show_progress=True if self.debug_level > 0 else False,
                force_text=True,
                margins=0               # Use full page
            )
            
            # Save markdown to file
            with open(self.md_path, 'w', encoding='utf-8') as f:
                f.write(self.markdown_content)
            
            debug(f"Markdown saved to {self.md_path}", 1, self.debug_level)
            return self.markdown_content
            
        except Exception as e:
            debug(f"Error extracting text: {str(e)}", 0, self.debug_level)
            raise
    
    def detect_document_structure(self) -> Dict[str, Any]:
        """Detect document structure including title, organization name, etc."""
        debug("Detecting document structure", 2, self.debug_level)
        
        # This is a simplified implementation. In a real application,
        # you might use regex patterns or NLP to extract organization name, etc.
        structure = {
            "title": self.title,
            "org_name": self.org_name,
        }
        
        # Try to extract title from first heading if available
        lines = self.markdown_content.split('\n')
        for line in lines[:20]:  # Look at first 20 lines only
            if line.startswith('# '):
                structure["title"] = line.replace('# ', '').strip()
                break
        
        return structure
    
    def create_chunks(self) -> List[Document]:
        """Split the markdown content into semantically meaningful chunks"""
        debug("Creating chunks from markdown content", 1, self.debug_level)
        
        # Check if chunks file already exists and not forcing recreation
        if os.path.exists(self.chunks_path) and not self.force:
            debug(f"Chunks file already exists at {self.chunks_path}. Loading...", 1, self.debug_level)
            # We need to recreate the Document objects from the saved chunks
            with open(self.chunks_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Simple detection to see if we have chunks already
                if "--- CHUNK 1 ---" in content:
                    debug("Detected saved chunks, loading them instead of recreating", 1, self.debug_level)
                    # We need to parse the saved chunks into Document objects
                    return []  # For now, just return empty list to signal skip chunking
        
        # Define header splitter to identify sections
        headers_to_split_on = [
            ("#", "header1"),
            ("##", "header2"),
            ("###", "header3"),
            ("####", "header4"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        # First split based on headers
        header_splits = markdown_splitter.split_text(self.markdown_content)
        
        # Then use recursive character splitter for further splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            length_function=len,
        )
        
        all_chunks = []
        # Process each header section
        for doc in header_splits:
            # Get header hierarchy information
            metadata = doc.metadata.copy()
            content = doc.page_content
            
            # Detect if this chunk is a table
            is_table = self._is_table_content(content)
            
            # For tables, keep them as one chunk regardless of size
            if is_table:
                metadata.update({
                    "source": self.pdf_path,
                    "filename": self.pdf_name,
                    "title": self.title,
                    "org_name": self.org_name,
                    "chunk_type": "table"
                })
                all_chunks.append(Document(page_content=content, metadata=metadata))
            else:
                # For text, split into smaller chunks
                smaller_chunks = text_splitter.create_documents([content], [metadata])
                for chunk in smaller_chunks:
                    # Enrich metadata
                    chunk.metadata.update({
                        "source": self.pdf_path,
                        "filename": self.pdf_name,
                        "title": self.title,
                        "org_name": self.org_name,
                        "chunk_type": "text"
                    })
                
                all_chunks.extend(smaller_chunks)
        
        # Enrich with page numbers
        self._enrich_with_page_numbers(all_chunks)
        
        # Save chunks to file for inspection
        with open(self.chunks_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(all_chunks):
                f.write(f"--- CHUNK {i+1} ---\n")
                f.write(f"Content:\n{chunk.page_content}\n\n")
                f.write(f"Metadata:\n{json.dumps(chunk.metadata, indent=2)}\n\n")
        
        debug(f"Created {len(all_chunks)} chunks, saved to {self.chunks_path}", 1, self.debug_level)
        self.chunks = all_chunks
        return all_chunks
    
    def _is_table_content(self, content: str) -> bool:
        """Detect if content is primarily a table"""
        # Simple heuristic: Count pipe characters and check if they appear frequently
        pipe_count = content.count('|')
        lines = content.split('\n')
        if pipe_count > 5 and pipe_count / len(lines) > 0.5:
            return True
        return False
    
    def _enrich_with_page_numbers(self, chunks: List[Document]) -> None:
        """Add page numbers to chunks based on PDF content"""
        debug("Enriching chunks with page numbers", 2, self.debug_level)
        
        if self.doc:
            total_pages = self.doc.page_count
        else:
            # Fallback if we couldn't open the PDF
            total_pages = 1
            debug("Could not get page count, using total_pages=1", 1, self.debug_level)
        
        # Look for page markers in the text
        page_markers = {}
        for i, chunk in enumerate(chunks):
            # Look for patterns that might indicate page numbers in the markdown
            content = chunk.page_content
            
            # Look for common page number formats like "Page X of Y" or just page numbers
            page_matches = re.findall(r"Page\s+(\d+)(?:\s+of\s+\d+)?", content)
            if page_matches:
                page_markers[i] = int(page_matches[0])
                continue
                
            # If no explicit marker, use position-based estimation
            if not page_markers.get(i):
                # Simple estimation based on chunk position in list
                estimated_page = min(1 + (i * total_pages // len(chunks)), total_pages)
                page_markers[i] = estimated_page
        
        # Apply page numbers to chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata["page_number"] = page_markers.get(i, 1)
    
    def process(self) -> List[Document]:
        """Run the processing pipeline for a single PDF file"""
        debug(f"Starting PDF processing for {self.pdf_path}", 1, self.debug_level)
        
        # Extract text
        self.extract_text()
        
        # Detect document structure
        doc_structure = self.detect_document_structure()
        self.title = doc_structure["title"]
        self.org_name = doc_structure["org_name"]
        
        # Create chunks
        self.create_chunks()
        
        # Return chunks
        return self.chunks


class MultiplePDFProcessor:
    """Processes multiple PDF files and creates a single vector database with client reuse"""
    
    def __init__(self, source_dir: str, output_dir: str = "./chromadb", debug_level: int = 1, force: bool = False):
        """Initialize the multiple PDF processor"""
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.debug_level = debug_level
        self.force = force
        
        # Initialize database client and embeddings once
        self.client = None
        self.embeddings = None
        self.vectordb = None
        
    def get_pdf_files(self) -> List[str]:
        """Get all PDF files in the source directory"""
        pdf_files = []
        
        # Check if source directory exists
        if not os.path.exists(self.source_dir):
            debug(f"Source directory not found: {self.source_dir}", 0, self.debug_level)
            return pdf_files
        
        # Get all PDF files
        for filename in os.listdir(self.source_dir):
            if filename.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(self.source_dir, filename))
        
        debug(f"Found {len(pdf_files)} PDF files in {self.source_dir}", 1, self.debug_level)
        return pdf_files
    
    def initialize_database(self) -> bool:
        """Initialize the vector database client and embeddings"""
        debug("Initializing vector database client and embeddings", 1, self.debug_level)
        
        # Check if environment variable for API key is set
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            debug("Warning: GOOGLE_API_KEY environment variable not found", 0, self.debug_level)
            debug("Set your API key with: export GOOGLE_API_KEY=your_key_here", 0, self.debug_level)
            return False
        
        try:
            # Initialize embedding function
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key
            )
            
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Clean existing database if force flag is set
            if self.force and os.path.exists(self.output_dir) and os.listdir(self.output_dir):
                debug(f"Force flag set, removing existing database at {self.output_dir}", 1, self.debug_level)
                import shutil
                for item in os.listdir(self.output_dir):
                    item_path = os.path.join(self.output_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=self.output_dir)
            
            # Check if our collection already exists
            collection_name = "rag_collection"
            if collection_name in [col.name for col in self.client.list_collections()]:
                debug(f"Found existing '{collection_name}' collection", 1, self.debug_level)
                # Connect to existing collection
                self.vectordb = Chroma(
                    client=self.client,
                    collection_name=collection_name,
                    embedding_function=self.embeddings
                )
            else:
                debug("No existing collection found, will create one when adding documents", 1, self.debug_level)
                # We'll create the collection when adding the first documents
                self.vectordb = None
            
            return True
            
        except Exception as e:
            debug(f"Error initializing database: {str(e)}", 0, self.debug_level)
            if self.debug_level >= 2:
                import traceback
                traceback.print_exc()
            return False
    
    def add_chunks_to_database(self, chunks: List[Document]) -> bool:
        """Add chunks to the vector database"""
        if not chunks:
            debug("No chunks to add", 1, self.debug_level)
            return True
        
        try:
            debug(f"Adding {len(chunks)} chunks to vector database", 1, self.debug_level)
            
            # Generate UUIDs for the chunks
            from uuid import uuid4
            chunk_ids = [str(uuid4()) for _ in range(len(chunks))]
            
            # If vectordb hasn't been created yet, create it with the first batch
            if self.vectordb is None:
                debug("Creating new ChromaDB collection", 1, self.debug_level)
                
                # Get or create the collection
                collection = self.client.get_or_create_collection("rag_collection")
                
                self.vectordb = Chroma(
                    client=self.client,
                    collection_name="rag_collection",
                    embedding_function=self.embeddings
                )
                
                # Add documents with explicit IDs
                self.vectordb.add_documents(documents=chunks, ids=chunk_ids)
            else:
                # Process in batches to avoid memory issues
                batch_size = 100
                total_batches = (len(chunks) + batch_size - 1) // batch_size
                
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:min(i+batch_size, len(chunks))]
                    batch_ids = chunk_ids[i:min(i+batch_size, len(chunks))]
                    batch_num = i // batch_size + 1
                    debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)", 2, self.debug_level)
                    
                    # Add documents with explicit IDs
                    self.vectordb.add_documents(documents=batch, ids=batch_ids)
                    
                    debug(f"Batch {batch_num} processed", 2, self.debug_level)
            
            return True
            
        except Exception as e:
            debug(f"Error adding chunks to database: {str(e)}", 0, self.debug_level)
            if self.debug_level >= 2:
                import traceback
                traceback.print_exc()
            return False
    
    def process_all(self) -> None:
        """Process all PDF files and add them to the vector database"""
        # Get PDF files
        pdf_files = self.get_pdf_files()
        if not pdf_files:
            debug(f"No PDF files found in {self.source_dir}", 0, self.debug_level)
            return
        
        # Initialize database
        if not self.initialize_database():
            debug("Failed to initialize database", 0, self.debug_level)
            return
        
        # Process each PDF file and add to database
        success_count = 0
        
        for i, pdf_file in enumerate(pdf_files):
            try:
                debug(f"Processing file {i+1}/{len(pdf_files)}: {pdf_file}", 1, self.debug_level)
                processor = PDFProcessor(pdf_file, self.debug_level, self.force)
                chunks = processor.process()
                
                if chunks:
                    if self.add_chunks_to_database(chunks):
                        debug(f"Added {len(chunks)} chunks from {pdf_file} to database", 1, self.debug_level)
                        success_count += 1
                else:
                    debug(f"No chunks generated for {pdf_file}", 1, self.debug_level)
            except Exception as e:
                debug(f"Error processing {pdf_file}: {str(e)}", 0, self.debug_level)
                if self.debug_level >= 2:
                    import traceback
                    traceback.print_exc()
        
        debug(f"PDF processing complete: {success_count}/{len(pdf_files)} files processed successfully", 1, self.debug_level)


def main():
    """Main entry point for the script"""
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert PDF files to vector database for RAG")
    parser.add_argument("--input", default="./sources", help="Directory containing PDF files to process (default: ./sources)")
    parser.add_argument("--output", default="./chromadb", help="Output directory for ChromaDB (default: ./chromadb)")
    parser.add_argument("--debug", type=int, choices=[0, 1, 2], default=1, help="Debug level: 0=none, 1=steps, 2=verbose (default: 1)")
    parser.add_argument("--force", action="store_true", help="Force recreation of all outputs")
    
    args = parser.parse_args()
    
    # Check if source directory exists
    if not os.path.exists(args.input):
        print(f"Error: Source directory not found: {args.input}", file=sys.stderr)
        print(f"Creating directory: {args.input}", file=sys.stderr)
        os.makedirs(args.input, exist_ok=True)
        return 1
    
    try:
        # Create processor and run
        processor = MultiplePDFProcessor(args.input, args.output, args.debug, args.force)
        processor.process_all()
        return 0
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        if args.debug >= 2:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())