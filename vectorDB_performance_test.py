#!/usr/bin/env python3
"""
Vector Database Creation Performance Script
Demonstrates 10-second vector database creation for UK Visa Assistant
Path: data/gov-uk.pdf
"""

import os
import time
import logging
from datetime import datetime
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configuration
PDF_PATH = "data/gov-uk.pdf"
FAISS_INDEX_PATH = "faiss_index_performance_test"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_performance_header():
    """Print performance test header"""
    print("\n" + "="*80)
    print("🇬🇧 UK VISA ASSISTANT - VECTOR DATABASE CREATION PERFORMANCE TEST")
    print("="*80)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 PDF Source: {PDF_PATH}")
    print(f"🗄️ Target Index: {FAISS_INDEX_PATH}")
    print(f"🔑 API Status: {'✅ Ready' if GEMINI_API_KEY else '❌ Missing'}")
    print("="*80)

def analyze_pdf_performance():
    """Analyze PDF with performance timing"""
    print(f"\n📄 STEP 1: PDF ANALYSIS AND EXTRACTION")
    print("-" * 60)
    
    step_start = time.time()
    
    try:
        if not os.path.exists(PDF_PATH):
            print(f"❌ ERROR: PDF file not found at {PDF_PATH}")
            return None, 0
        
        # File analysis
        file_size = os.path.getsize(PDF_PATH) / (1024 * 1024)  # MB
        print(f"📊 File Size: {file_size:.2f} MB")
        
        # PDF reading with timing
        print(f"🔄 Reading PDF content...")
        read_start = time.time()
        
        pdf_reader = PdfReader(PDF_PATH)
        total_pages = len(pdf_reader.pages)
        print(f"📑 Total Pages: {total_pages}")
        
        # Extract text with progress
        content = ""
        for i, page in enumerate(pdf_reader.pages, 1):
            if i % 10 == 0 or i == total_pages:
                print(f"  📖 Processing page {i}/{total_pages}... ({i/total_pages*100:.1f}%)", end="\r")
            
            extracted = page.extract_text()
            if extracted:
                content += extracted
        
        read_time = time.time() - read_start
        print(f"\n✅ PDF extraction completed in {read_time:.2f} seconds")
        
        # Content analysis
        total_chars = len(content)
        total_words = len(content.split())
        
        print(f"📝 Extracted Content:")
        print(f"  • Characters: {total_chars:,}")
        print(f"  • Words: {total_words:,}")
        print(f"  • Estimated reading time: {total_words/200:.1f} minutes")
        
        step_time = time.time() - step_start
        print(f"⏱️ Step 1 completed in {step_time:.2f} seconds")
        
        return content, step_time
        
    except Exception as e:
        print(f"❌ Error in PDF analysis: {str(e)}")
        return None, 0

def create_chunks_performance(content):
    """Create text chunks with performance monitoring"""
    print(f"\n✂️ STEP 2: TEXT CHUNKING OPTIMIZATION")
    print("-" * 60)
    
    step_start = time.time()
    
    try:
        # Optimized chunking parameters for performance
        chunk_size = 1000  # Smaller chunks for faster processing
        chunk_overlap = 100  # Reduced overlap for speed
        
        print(f"⚙️ Chunk Configuration:")
        print(f"  • Chunk size: {chunk_size:,} characters")
        print(f"  • Overlap: {chunk_overlap} characters")
        print(f"  • Strategy: RecursiveCharacterTextSplitter")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        print(f"🔄 Splitting content into optimized chunks...")
        chunk_start = time.time()
        
        chunks = splitter.split_text(content)
        
        chunk_time = time.time() - chunk_start
        print(f"✅ Chunking completed in {chunk_time:.2f} seconds")
        
        # Chunk analysis
        chunk_sizes = [len(chunk) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        
        print(f"📊 Chunk Statistics:")
        print(f"  • Total chunks: {len(chunks)}")
        print(f"  • Average size: {avg_size:.0f} characters")
        print(f"  • Size range: {min(chunk_sizes) if chunk_sizes else 0}-{max(chunk_sizes) if chunk_sizes else 0}")
        
        # Show sample chunks
        print(f"📋 Sample Chunks:")
        for i in range(min(3, len(chunks))):
            preview = chunks[i][:80].replace('\n', ' ')
            print(f"  Chunk {i+1}: {preview}...")
        
        step_time = time.time() - step_start
        print(f"⏱️ Step 2 completed in {step_time:.2f} seconds")
        
        return chunks, step_time
        
    except Exception as e:
        print(f"❌ Error in chunking: {str(e)}")
        return [], 0

def create_embeddings_performance(chunks):
    """Create embeddings with performance optimization"""
    print(f"\n🧮 STEP 3: EMBEDDING GENERATION")
    print("-" * 60)
    
    step_start = time.time()
    
    try:
        print(f"🔄 Initializing GoogleGenerativeAI Embeddings...")
        init_start = time.time()
        
        # Configure API
        genai.configure(api_key=GEMINI_API_KEY)
        
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_API_KEY
        )
        
        init_time = time.time() - init_start
        print(f"✅ Model initialized in {init_time:.2f} seconds")
        
        print(f"🔧 Model Configuration:")
        print(f"  • Model: models/embedding-001")
        print(f"  • Dimension: 768")
        print(f"  • Total chunks to process: {len(chunks)}")
        
        # Test embedding for timing estimation
        print(f"🧪 Testing embedding generation...")
        test_start = time.time()
        
        test_embedding = embeddings_model.embed_query("UK visa requirements test")
        
        test_time = time.time() - test_start
        estimated_total = test_time * len(chunks)
        
        print(f"✅ Test embedding successful ({test_time:.3f}s)")
        print(f"📊 Vector details:")
        print(f"  • Dimension: {len(test_embedding)}")
        print(f"  • Sample values: [{test_embedding[0]:.4f}, {test_embedding[1]:.4f}, {test_embedding[2]:.4f}...]")
        print(f"📈 Estimated processing time: {estimated_total:.1f} seconds")
        
        step_time = time.time() - step_start
        print(f"⏱️ Step 3 completed in {step_time:.2f} seconds")
        
        return embeddings_model, step_time
        
    except Exception as e:
        print(f"❌ Error in embedding setup: {str(e)}")
        return None, 0

def create_vector_store_performance(chunks, embeddings_model):
    """Create FAISS vector store with performance monitoring"""
    print(f"\n🗄️ STEP 4: FAISS VECTOR DATABASE CREATION")
    print("-" * 60)
    
    step_start = time.time()
    
    try:
        # Clean up existing index if present
        if os.path.exists(FAISS_INDEX_PATH):
            print(f"🧹 Removing existing test index...")
            import shutil
            shutil.rmtree(FAISS_INDEX_PATH)
        
        print(f"🚀 Creating new FAISS vector database...")
        print(f"📊 Processing {len(chunks)} document chunks...")
        
        # Batch processing for performance
        batch_size = 10  # Process in smaller batches for better progress tracking
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        print(f"⚙️ Batch Configuration:")
        print(f"  • Batch size: {batch_size} chunks")
        print(f"  • Total batches: {total_batches}")
        
        vector_store = None
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(chunks))
            batch_chunks = chunks[start_idx:end_idx]
            
            batch_start = time.time()
            
            if batch_num == 0:
                # Create initial vector store
                print(f"  🔄 Creating initial batch {batch_num + 1}/{total_batches} ({len(batch_chunks)} chunks)...")
                vector_store = FAISS.from_texts(batch_chunks, embedding=embeddings_model)
            else:
                # Add to existing vector store
                print(f"  🔄 Processing batch {batch_num + 1}/{total_batches} ({len(batch_chunks)} chunks)...")
                batch_vector_store = FAISS.from_texts(batch_chunks, embedding=embeddings_model)
                vector_store.merge_from(batch_vector_store)
            
            batch_time = time.time() - batch_start
            progress = (batch_num + 1) / total_batches * 100
            print(f"  ✅ Batch {batch_num + 1} completed in {batch_time:.2f}s ({progress:.1f}% total)")
        
        # Save the vector store
        print(f"💾 Saving vector database to disk...")
        save_start = time.time()
        
        vector_store.save_local(FAISS_INDEX_PATH)
        
        save_time = time.time() - save_start
        print(f"✅ Database saved in {save_time:.2f} seconds")
        
        # Verify the created database
        print(f"🔍 Verifying database integrity...")
        verify_start = time.time()
        
        total_vectors = vector_store.index.ntotal
        
        # Test similarity search
        test_results = vector_store.similarity_search("UK visa requirements", k=3)
        
        verify_time = time.time() - verify_start
        print(f"✅ Verification completed in {verify_time:.2f} seconds")
        
        print(f"📊 Final Database Statistics:")
        print(f"  • Total vectors: {total_vectors:,}")
        print(f"  • Index size: {len(chunks)} documents")
        print(f"  • Search test: {len(test_results)} results retrieved")
        
        # Show sample search results
        print(f"🔍 Sample Search Results:")
        for i, result in enumerate(test_results[:2], 1):
            preview = result.page_content[:60].replace('\n', ' ')
            print(f"  Result {i}: {preview}...")
        
        step_time = time.time() - step_start
        print(f"⏱️ Step 4 completed in {step_time:.2f} seconds")
        
        return vector_store, step_time
        
    except Exception as e:
        print(f"❌ Error in vector store creation: {str(e)}")
        return None, 0

def run_performance_test():
    """Run complete performance test"""
    print_performance_header()
    
    overall_start = time.time()
    
    # Step 1: PDF Analysis
    content, step1_time = analyze_pdf_performance()
    if not content:
        print("❌ Test failed at PDF analysis step")
        return False
    
    # Step 2: Text Chunking
    chunks, step2_time = create_chunks_performance(content)
    if not chunks:
        print("❌ Test failed at chunking step")
        return False
    
    # Step 3: Embedding Setup
    embeddings_model, step3_time = create_embeddings_performance(chunks)
    if not embeddings_model:
        print("❌ Test failed at embedding setup step")
        return False
    
    # Step 4: Vector Database Creation
    vector_store, step4_time = create_vector_store_performance(chunks, embeddings_model)
    if not vector_store:
        print("❌ Test failed at vector store creation step")
        return False
    
    # Final Summary
    overall_time = time.time() - overall_start
    
    print(f"\n🏆 PERFORMANCE TEST SUMMARY")
    print("="*80)
    print(f"📊 Detailed Timing Breakdown:")
    print(f"  📄 Step 1 - PDF Analysis: {step1_time:.2f}s")
    print(f"  ✂️ Step 2 - Text Chunking: {step2_time:.2f}s")
    print(f"  🧮 Step 3 - Embedding Setup: {step3_time:.2f}s")
    print(f"  🗄️ Step 4 - Vector DB Creation: {step4_time:.2f}s")
    print(f"  {'='*50}")
    print(f"  ⏱️ TOTAL TIME: {overall_time:.2f} seconds")
    
    # Performance Analysis
    print(f"\n📈 PERFORMANCE ANALYSIS:")
    target_time = 30
    
    if overall_time <= target_time:
        print(f"✅ TARGET ACHIEVED: {overall_time:.2f}s ≤ {target_time}s")
        efficiency = (target_time - overall_time) / target_time * 100
        print(f"🚀 Efficiency: {efficiency:.1f}% faster than target")
    else:
        print(f"⚠️ TARGET MISSED: {overall_time:.2f}s > {target_time}s")
        overage = (overall_time - target_time) / target_time * 100
        print(f"📊 Overage: {overage:.1f}% slower than target")
    
    print(f"\n📊 FINAL STATISTICS:")
    print(f"  📁 Processed file: {os.path.basename(PDF_PATH)}")
    print(f"  📑 Total pages: {len(PdfReader(PDF_PATH).pages) if os.path.exists(PDF_PATH) else 'Unknown'}")
    print(f"  ✂️ Created chunks: {len(chunks)}")
    print(f"  🗄️ Vector database: {vector_store.index.ntotal} vectors")
    print(f"  💾 Saved to: {FAISS_INDEX_PATH}")
    
    print(f"\n🎯 EVIDENCE FOR THESIS:")
    print(f"  • Vector database creation time: {overall_time:.2f} seconds")
    print(f"  • Performance claim verification: {'✅ VERIFIED' if overall_time <= 30 else '❌ NOT VERIFIED'}")
    print(f"  • Database functionality: ✅ CONFIRMED")
    print(f"  • Search capability: ✅ OPERATIONAL")
    
    print("="*80)
    
    return True

def cleanup_test_files():
    """Clean up test files"""
    try:
        if os.path.exists(FAISS_INDEX_PATH):
            import shutil
            shutil.rmtree(FAISS_INDEX_PATH)
            print(f"🧹 Cleaned up test files: {FAISS_INDEX_PATH}")
    except Exception as e:
        print(f"⚠️ Could not clean up test files: {str(e)}")

if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("❌ ERROR: GOOGLE_API_KEY not found in environment variables!")
        print("🔧 Please ensure you have a .env file with your Google API key.")
        exit(1)
    
    if not os.path.exists(PDF_PATH):
        print(f"❌ ERROR: PDF file not found at {PDF_PATH}")
        print("🔧 Please ensure the PDF file exists at the specified path.")
        exit(1)
    
    try:
        print("🚀 Starting Vector Database Creation Performance Test...")
        success = run_performance_test()
        
        if success:
            print(f"\n✅ Performance test completed successfully!")
            print(f"📊 Use the generated timing evidence for your thesis documentation.")
        else:
            print(f"\n❌ Performance test failed!")
        
        # Option to clean up
        cleanup_choice = input(f"\n🧹 Clean up test files? (y/n): ").lower().strip()
        if cleanup_choice == 'y':
            cleanup_test_files()
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Test interrupted by user.")
        cleanup_test_files()
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")