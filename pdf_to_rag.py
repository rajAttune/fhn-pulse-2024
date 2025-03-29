#!/usr/bin/env python3
"""
Enhanced PDF to Vector Database Converter for RAG with Image Processing

This script extracts text and images from multiple PDF files, preserves document structure,
creates semantic chunks, processes images through Gemini for descriptions and insights,
and stores everything in a ChromaDB vector database for RAG.

Usage:
    python enhanced_pdf_to_rag.py [--input SOURCES_DIR] [--output CHROMADB_DIR] [--debug LEVEL] [--force]
"""

import os
import sys
import argparse
import re
import base64
from typing import List, Dict, Any, Optional, Tuple, Set
import json
from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv
import hashlib

# PyMuPDF for PDF extraction
try:
    import fitz  # PyMuPDF
    import pymupdf4llm
except ImportError:
    print("Error: PyMuPDF and PyMuPDF4LLM are required. Install with: pip install pymupdf pymupdf4llm", file=sys.stderr)
    sys.exit(1)

# LangChain Gemini API for image processing
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
except ImportError:
    print("Error: LangChain Google Generative AI is required. Install with: pip install langchain-google-genai", file=sys.stderr)
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
        
        # Document for page count and image extraction
        try:
            self.doc = fitz.open(self.pdf_path)
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
            # For now, just return empty list to indicate chunks already exist
            # (We'll actually load chunks in the process method)
            return []
        
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
                    "chunk_type": "text"  # Add chunk type metadata
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
                        "chunk_type": "text"  # Add chunk type metadata
                    })
                
                all_chunks.extend(smaller_chunks)
        
        # Enrich with page numbers
        self._enrich_with_page_numbers(all_chunks)
        
        self.chunks = all_chunks
        return all_chunks
    
    def _is_table_content(self, content: str) -> bool:
        """Detect if content is primarily a table (for special handling, not for metadata)"""
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
            total_pages = len(self.doc)
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
    
    def write_chunks_to_file(self, chunks: List[Document]) -> None:
        """Write chunks to a text file for inspection"""
        debug(f"Writing {len(chunks)} chunks to {self.chunks_path}", 1, self.debug_level)
        
        with open(self.chunks_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                chunk_type = chunk.metadata.get("chunk_type", "unknown")
                f.write(f"--- {chunk_type.upper()} CHUNK {i+1} ---\n")
                f.write(f"Content:\n{chunk.page_content}\n\n")
                f.write(f"Metadata:\n{json.dumps(chunk.metadata, indent=2)}\n\n")
    
    def load_existing_chunks(self) -> List[Document]:
        """Load existing chunks from the chunks file"""
        chunks = []
        if os.path.exists(self.chunks_path):
            debug(f"Loading existing chunks from {self.chunks_path}", 1, self.debug_level)
            # In a real application, you'd parse the chunks file to recreate Document objects
            # This is a simplified implementation that returns an empty list
            # indicating that chunks should be fetched from the database
            return []
        return chunks
    
    def process(self) -> List[Document]:
        """Run the processing pipeline for a single PDF file"""
        debug(f"Starting PDF processing for {self.pdf_path}", 1, self.debug_level)
        
        # Extract text
        self.extract_text()
        
        # Detect document structure
        doc_structure = self.detect_document_structure()
        self.title = doc_structure["title"]
        self.org_name = doc_structure["org_name"]
        
        # Create or load chunks
        if os.path.exists(self.chunks_path) and not self.force:
            self.chunks = self.load_existing_chunks()
        else:
            self.chunks = self.create_chunks()
        
        # Return chunks
        return self.chunks


class ImageProcessor:
    """Processes images from PDF files and generates descriptions using Gemini"""
    
    def __init__(self, pdf_path: str, output_dir: str = None, debug_level: int = 1, force: bool = False):
        """Initialize the image processor for a single PDF file"""
        self.pdf_path = pdf_path
        self.debug_level = debug_level
        self.force = force
        
        # Setup output paths
        self.pdf_dir = os.path.dirname(os.path.abspath(pdf_path))
        self.pdf_name = os.path.basename(pdf_path)
        self.base_name = os.path.splitext(self.pdf_name)[0]
        
        # Create output directory for images if not provided
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.join(self.pdf_dir, f"{self.base_name}_images")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Path for image metadata and parent PDF chunks file
        self.images_metadata_path = os.path.join(self.output_dir, "images_metadata.json")
        self.chunks_path = os.path.join(self.pdf_dir, f"{self.base_name}_chunks.txt")
        
        # Document for image extraction
        try:
            self.doc = fitz.open(self.pdf_path)
            self.total_pages = len(self.doc)
        except Exception as e:
            debug(f"Error opening PDF with PyMuPDF: {str(e)}", 0, self.debug_level)
            self.doc = None
            self.total_pages = 0
        
        # Initialize image data structures
        self.images_metadata = []
        self.image_chunks = []
        
        # Set up Gemini for image processing
        self._setup_gemini()
    
    def _setup_gemini(self):
        """Set up the Gemini API for image processing using LangChain"""
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            debug("Warning: GOOGLE_API_KEY environment variable not found", 0, self.debug_level)
            debug("Set your API key with: export GOOGLE_API_KEY=your_key_here", 0, self.debug_level)
            return False
        
        # Initialize Gemini model through LangChain
        try:
            # Using the modern recommended approach without the deprecated parameter
            self.model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=api_key
            )
            return True
        except Exception as e:
            debug(f"Error initializing Gemini model: {str(e)}", 0, self.debug_level)
            self.model = None
            return False
    
    def _should_skip_image(self, img_info: Dict) -> bool:
        """Determine if an image should be skipped (e.g., too small, likely a logo)"""
        # Skip if image is too small (likely decorative or a logo)
        min_width = 100
        min_height = 100
        if img_info["width"] < min_width or img_info["height"] < min_height:
            return True
        
        # Skip if aspect ratio is extreme (likely a line or decoration)
        aspect_ratio = img_info["width"] / max(img_info["height"], 1)
        if aspect_ratio > 10 or aspect_ratio < 0.1:
            return True
        
        return False
    
    def _get_image_hash(self, image_bytes: bytes) -> str:
        """Generate a hash for image deduplication"""
        return hashlib.md5(image_bytes).hexdigest()
    
    def _get_page_text(self, page_num: int) -> str:
        """Get the text content from a specific page"""
        if not self.doc:
            return ""
        
        try:
            page = self.doc[page_num]
            return page.get_text()
        except Exception as e:
            debug(f"Error getting text from page {page_num}: {str(e)}", 0, self.debug_level)
            return ""
    
    def _extract_section_heading(self, page_text: str) -> str:
        """Extract section heading from page text"""
        # Simple implementation to find potential headings
        # Look for lines that might be headings (uppercase, or followed by newlines)
        lines = page_text.split('\n')
        for line in lines[:10]:  # Look at first 10 lines
            line = line.strip()
            if len(line) > 0 and len(line) < 100:  # Reasonable heading length
                if line.isupper() or line.istitle():
                    return line
        
        return "Unknown Section"
    
    def extract_and_process_images(self) -> List[Document]:
        """Extract images from PDF, process them with Gemini, and create chunks"""
        debug(f"Extracting and processing images from {self.pdf_path}", 1, self.debug_level)
        
        # Check if images metadata already exists and not forcing recreation
        if os.path.exists(self.images_metadata_path) and not self.force:
            debug(f"Images metadata already exists at {self.images_metadata_path}", 1, self.debug_level)
            # Try to load previously processed image chunks
            return self.load_existing_image_chunks()
        
        if not self.doc or not self.model:
            debug("Document or Gemini model not available, skipping image processing", 0, self.debug_level)
            return []
        
        # Set to track processed image hashes for deduplication
        processed_hashes = set()
        
        # Extract and process images page by page
        for page_num in range(self.total_pages):
            debug(f"Processing images on page {page_num + 1}/{self.total_pages}", 2, self.debug_level)
            
            # Get the page
            page = self.doc[page_num]
            
            # Get page text for context
            page_text = page.get_text()
            section_heading = self._extract_section_heading(page_text)
            
            # Get image list from page
            image_list = page.get_images(full=True)
            
            # Skip if no images on page
            if not image_list:
                continue
            
            # Process each image on the page
            for img_index, img in enumerate(image_list):
                try:
                    # Get basic image info
                    xref = img[0]
                    base_image = self.doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Generate image hash for deduplication
                    image_hash = self._get_image_hash(image_bytes)
                    
                    # Skip if already processed (duplicate)
                    if image_hash in processed_hashes:
                        debug(f"Skipping duplicate image on page {page_num + 1}", 2, self.debug_level)
                        continue
                    
                    # Extract image position data
                    pix = fitz.Pixmap(self.doc, xref)
                    
                    # Create image info dictionary
                    img_info = {
                        "page_num": page_num + 1,
                        "img_index": img_index,
                        "width": pix.width,
                        "height": pix.height,
                        "image_hash": image_hash,
                        "image_format": base_image["ext"],
                        "section_heading": section_heading
                    }
                    
                    # Skip small or decorative images
                    if self._should_skip_image(img_info):
                        debug(f"Skipping small/decorative image on page {page_num + 1}", 2, self.debug_level)
                        continue
                    
                    # Save image to file
                    image_filename = f"page{page_num + 1}_img{img_index}.{base_image['ext']}"
                    image_path = os.path.join(self.output_dir, image_filename)
                    
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    
                    img_info["image_path"] = image_path
                    
                    # Process image with Gemini
                    img_description = self._process_image_with_gemini(image_bytes, page_text, img_info)
                    
                    if img_description:
                        # Add description to image info
                        img_info["description"] = img_description["description"]
                        img_info["potential_questions"] = img_description["potential_questions"]
                        
                        # Add to metadata list
                        self.images_metadata.append(img_info)
                        
                        # Create document chunk for this image
                        image_doc = Document(
                            page_content=self._format_image_content(img_info),
                            metadata={
                                "source": self.pdf_path,
                                "filename": self.pdf_name,
                                "page_number": page_num + 1,
                                "image_path": image_path,
                                "section_heading": section_heading,
                                "width": pix.width,
                                "height": pix.height,
                                "chunk_type": "image"  # Mark as image chunk
                            }
                        )
                        
                        self.image_chunks.append(image_doc)
                        
                        # Add to processed set
                        processed_hashes.add(image_hash)
                        
                except Exception as e:
                    debug(f"Error processing image on page {page_num + 1}: {str(e)}", 0, self.debug_level)
                    if self.debug_level >= 2:
                        import traceback
                        traceback.print_exc()
        
        # Save image metadata to file
        with open(self.images_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.images_metadata, f, indent=2)
        
        debug(f"Processed {len(self.image_chunks)} unique images", 1, self.debug_level)
        
        return self.image_chunks
    
    def load_existing_image_chunks(self) -> List[Document]:
        """Load existing image chunks based on saved metadata"""
        if not os.path.exists(self.images_metadata_path):
            return []
        
        try:
            with open(self.images_metadata_path, 'r', encoding='utf-8') as f:
                image_metadata = json.load(f)
            
            image_chunks = []
            for img_info in image_metadata:
                # Create document chunk for this image
                image_doc = Document(
                    page_content=self._format_image_content(img_info),
                    metadata={
                        "source": self.pdf_path,
                        "filename": self.pdf_name,
                        "page_number": img_info.get("page_num", 1),
                        "image_path": img_info.get("image_path", ""),
                        "section_heading": img_info.get("section_heading", "Unknown Section"),
                        "width": img_info.get("width", 0),
                        "height": img_info.get("height", 0),
                        "chunk_type": "image"  # Mark as image chunk
                    }
                )
                image_chunks.append(image_doc)
            
            debug(f"Loaded {len(image_chunks)} image chunks from metadata", 1, self.debug_level)
            return image_chunks
            
        except Exception as e:
            debug(f"Error loading image chunks from metadata: {str(e)}", 0, self.debug_level)
            return []
    
    def _process_image_with_gemini(self, image_bytes: bytes, page_text: str, img_info: Dict) -> Dict:
        """Process an image using Gemini via LangChain to generate a description and potential questions"""
        try:
            debug(f"Processing image with Gemini on page {img_info['page_num']}", 2, self.debug_level)
            
            # Create a base64 representation of the image
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            
            # Create a prompt for Gemini that includes context
            prompt = f"""
            This is a figure or chart from page {img_info['page_num']} of a document.
            Section: {img_info['section_heading']}
            
            Please analyze this image thoroughly and provide:
            1. A detailed description of what the image shows, including any data, trends, or key information
            2. An interpretation of the image in the context of the surrounding text
            3. A list of the top 3-5 specific questions a reader might ask that this image would help answer
            
            Here is some context from the page text:
            {page_text[:1000]}...
            
            Format your response as a JSON object with these keys:
            - description: Your detailed description and interpretation
            - potential_questions: An array of specific questions this image might answer
            """
            
            # Create message with image for LangChain Gemini model
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{img_info['image_format']};base64,{base64_image}"
                        }
                    }
                ]
            )
            
            # Send to Gemini via LangChain
            response = self.model.invoke([message])
            
            # Process the response
            result_text = response.content
            
            # Parse the JSON response
            # Sometimes Gemini might wrap the JSON in markdown code blocks, so we need to handle that
            if "```json" in result_text:
                json_match = re.search(r"```json\n(.*?)\n```", result_text, re.DOTALL)
                if json_match:
                    result_text = json_match.group(1)
            
            try:
                result = json.loads(result_text)
                return result
            except:
                # Fallback: create a structured response from unstructured text
                debug("Failed to parse JSON from Gemini, creating structured response manually", 1, self.debug_level)
                
                # Extract description and questions using regex
                description = "Description unavailable"
                if "description" in result_text.lower():
                    desc_match = re.search(r"description[:\s]+(.*?)(?=potential|$)", result_text, re.DOTALL | re.IGNORECASE)
                    if desc_match:
                        description = desc_match.group(1).strip()
                
                questions = []
                questions_section = re.search(r"potential.questions.*?:(.*?)(?=$)", result_text, re.DOTALL | re.IGNORECASE)
                if questions_section:
                    # Extract questions that typically start with numbers or dashes
                    q_matches = re.findall(r"(?:^|\n)\s*(?:\d+[\.\)]*|[-*])\s*(.*?)(?=$|\n\s*(?:\d+[\.\)]*|[-*]))", 
                                          questions_section.group(1), re.DOTALL)
                    questions = [q.strip() for q in q_matches if q.strip()]
                
                return {
                    "description": description,
                    "potential_questions": questions
                }
                
        except Exception as e:
            debug(f"Error in Gemini processing: {str(e)}", 0, self.debug_level)
            if self.debug_level >= 2:
                import traceback
                traceback.print_exc()
            
            # Return empty results on error
            return {
                "description": f"[Image processing error: {str(e)}]",
                "potential_questions": []
            }
    
    def _format_image_content(self, img_info: Dict) -> str:
        """Format the image information into a searchable text chunk"""
        content = f"""# Image on Page {img_info['page_num']} - {img_info['section_heading']}

        ## Description
        {img_info['description']}

        ## Potential Questions This Image Answers
        """
        
        for i, question in enumerate(img_info['potential_questions']):
            content += f"{i+1}. {question}\n"
        
        content += f"\n[Source: Page {img_info['page_num']} of {self.pdf_name}]"
        
        return content
    
    def process(self) -> List[Document]:
        """Run the image processing pipeline for a single PDF file"""
        debug(f"Starting image processing for {self.pdf_path}", 1, self.debug_level)
        
        # Extract and process images
        image_chunks = self.extract_and_process_images()
        
        # Return image chunks
        return image_chunks


class EnhancedPDFProcessor:
    """Processes multiple PDF files, extracts text and images, and creates a single vector database"""
    
    def __init__(self, source_dir: str, output_dir: str = "./chromadb", debug_level: int = 1, force: bool = False):
        """Initialize the enhanced PDF processor"""
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.debug_level = debug_level
        self.force = force
        
        # Create directory for image output
        self.images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        
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
                    if os.path.isdir(item_path) and item != "images":
                        shutil.rmtree(item_path)
                    elif os.path.isfile(item_path):
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
        
        # Process each PDF file
        for i, pdf_file in enumerate(pdf_files):
            try:
                debug(f"Processing file {i+1}/{len(pdf_files)}: {pdf_file}", 1, self.debug_level)
                
                # Initialize processors
                text_processor = PDFProcessor(pdf_file, self.debug_level, self.force)
                
                # Process text chunks
                text_chunks = text_processor.process()
                
                # Process image chunks
                base_name = os.path.splitext(os.path.basename(pdf_file))[0]
                image_output_dir = os.path.join(self.images_dir, base_name)
                img_processor = ImageProcessor(pdf_file, image_output_dir, self.debug_level, self.force)
                image_chunks = img_processor.process()
                
                # Combine all chunks
                all_chunks = text_chunks + image_chunks
                
                # Write combined chunks to a single file
                if all_chunks:
                    text_processor.write_chunks_to_file(all_chunks)
                    
                    # Add all chunks to database
                    if self.add_chunks_to_database(all_chunks):
                        debug(f"Added {len(all_chunks)} chunks from {pdf_file} to database", 1, self.debug_level)
                    else:
                        debug(f"Failed to add chunks from {pdf_file} to database", 0, self.debug_level)
                else:
                    debug(f"No chunks generated for {pdf_file}", 1, self.debug_level)
                    
            except Exception as e:
                debug(f"Error processing {pdf_file}: {str(e)}", 0, self.debug_level)
                if self.debug_level >= 2:
                    import traceback
                    traceback.print_exc()


def main():
    """Main entry point for the script"""
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert PDF files to vector database for RAG with image support")
    parser.add_argument("--input", default="./sources", help="Directory containing PDF files to process (default: ./sources)")
    parser.add_argument("--output", default="./chromadb", help="Output directory for ChromaDB (default: ./chromadb)")
    parser.add_argument("--debug", type=int, choices=[0, 1, 2], default=2, help="Debug level: 0=none, 1=steps, 2=verbose (default: 2)")
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
        processor = EnhancedPDFProcessor(args.input, args.output, args.debug, args.force)
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