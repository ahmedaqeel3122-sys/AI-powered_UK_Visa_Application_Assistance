from flask import Flask, render_template, request, jsonify
import os
import time
import re
import logging
from datetime import datetime
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import pickle
import numpy as np

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('visa_assistant.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
PDF_PATH = "data/gov-uk.pdf"
FAISS_INDEX_PATH = "faiss_index"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Global variables
conversation_chain = None
pdf_processed = False
vector_store = None
embeddings_model = None
document_chunks = []
embedding_stats = {}

# Configure Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def print_banner():
    """Print a detailed banner with system information"""
    print("\n" + "="*80)
    print("🇬🇧  UK VISA AI ASSISTANT - DETAILED SYSTEM INITIALIZATION")
    print("="*80)
    print(f"⏰ Startup Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔑 API Key Status: {'✅ Configured' if GEMINI_API_KEY else '❌ Missing'}")
    print(f"📁 PDF Path: {PDF_PATH}")
    print(f"🗄️ FAISS Index Path: {FAISS_INDEX_PATH}")
    print(f"🌐 Flask Debug Mode: {'✅ Enabled' if app.debug else '❌ Disabled'}")
    print("="*80)

def print_pdf_analysis():
    """Analyze and print PDF details"""
    global document_chunks
    
    print("\n📄 PDF DOCUMENT ANALYSIS")
    print("-" * 50)
    
    try:
        if not os.path.exists(PDF_PATH):
            print(f"❌ PDF file not found at: {PDF_PATH}")
            return False
            
        # Get file size
        file_size = os.path.getsize(PDF_PATH) / (1024 * 1024)  # MB
        print(f"📊 File Size: {file_size:.2f} MB")
        
        # Read PDF and analyze
        pdf_reader = PdfReader(PDF_PATH)
        total_pages = len(pdf_reader.pages)
        print(f"📑 Total Pages: {total_pages}")
        
        # Extract content with progress
        content = ""
        char_count = 0
        
        for i, page in enumerate(pdf_reader.pages, 1):
            print(f"  📖 Processing page {i}/{total_pages}...", end="\r")
            extracted = page.extract_text()
            if extracted:
                content += extracted
                char_count += len(extracted)
        
        print(f"\n✅ Text Extraction Complete!")
        print(f"📝 Total Characters: {char_count:,}")
        print(f"📝 Total Words: {len(content.split()):,}")
        
        # Analyze content patterns
        print("\n🔍 CONTENT ANALYSIS:")
        visa_mentions = len(re.findall(r'\bvisa\b', content, re.IGNORECASE))
        requirement_mentions = len(re.findall(r'\brequirement\b', content, re.IGNORECASE))
        application_mentions = len(re.findall(r'\bapplication\b', content, re.IGNORECASE))
        
        print(f"  🔸 'Visa' mentions: {visa_mentions}")
        print(f"  🔸 'Requirement' mentions: {requirement_mentions}")
        print(f"  🔸 'Application' mentions: {application_mentions}")
        
        return content
        
    except Exception as e:
        print(f"❌ Error analyzing PDF: {str(e)}")
        return None

def print_chunking_process(content):
    """Detailed chunking process visualization"""
    global document_chunks
    
    print("\n✂️ TEXT CHUNKING PROCESS")
    print("-" * 50)
    
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        print(f"⚙️ Chunk Size: 10,000 characters")
        print(f"⚙️ Overlap: 1,000 characters")
        print(f"⚙️ Separator Strategy: Recursive (sentences → paragraphs → words)")
        
        print("🔄 Splitting content into chunks...")
        document_chunks = splitter.split_text(content)
        
        print(f"✅ Created {len(document_chunks)} chunks")
        
        # Analyze chunk statistics
        chunk_sizes = [len(chunk) for chunk in document_chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        min_size = min(chunk_sizes)
        max_size = max(chunk_sizes)
        
        print(f"📊 CHUNK STATISTICS:")
        print(f"  📏 Average size: {avg_size:.0f} characters")
        print(f"  📏 Minimum size: {min_size} characters")
        print(f"  📏 Maximum size: {max_size} characters")
        
        # Show sample chunks
        print(f"\n📋 SAMPLE CHUNKS:")
        for i in range(min(3, len(document_chunks))):
            preview = document_chunks[i][:100].replace('\n', ' ')
            print(f"  Chunk {i+1}: {preview}...")
            
        return True
        
    except Exception as e:
        print(f"❌ Error in chunking: {str(e)}")
        return False

def print_embedding_process():
    """Detailed embedding creation process"""
    global embeddings_model, embedding_stats
    
    print("\n🧮 VECTOR EMBEDDING PROCESS")
    print("-" * 50)
    
    try:
        # Initialize embeddings model
        print("🔄 Initializing GoogleGenerativeAI Embeddings...")
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=GEMINI_API_KEY
        )
        print("✅ Embeddings model initialized")
        
        print(f"🔧 Model: models/embedding-001")
        print(f"🔧 Dimension: 768 (standard for text-embedding-001)")
        print(f"🔧 Max Token Limit: 2048 tokens per chunk")
        
        # Test embedding on sample text
        print("\n🧪 Testing embedding generation...")
        sample_text = "UK visa application requirements"
        test_embedding = embeddings_model.embed_query(sample_text)
        
        print(f"✅ Test embedding successful")
        print(f"📊 Embedding vector length: {len(test_embedding)}")
        print(f"📊 Sample values: [{test_embedding[0]:.4f}, {test_embedding[1]:.4f}, {test_embedding[2]:.4f}...]")
        
        embedding_stats = {
            'model': 'models/embedding-001',
            'dimension': len(test_embedding),
            'total_chunks': len(document_chunks),
            'test_successful': True
        }
        
        return True
        
    except Exception as e:
        print(f"❌ Error in embedding setup: {str(e)}")
        return False

def print_faiss_operations():
    """Detailed FAISS database operations"""
    global vector_store
    
    print("\n🗄️ FAISS VECTOR DATABASE OPERATIONS")
    print("-" * 50)
    
    try:
        # Check if existing index exists
        if os.path.exists(FAISS_INDEX_PATH):
            print("🔍 Existing FAISS index detected")
            
            # Load and analyze existing index
            print("📥 Loading existing FAISS index...")
            vector_store = FAISS.load_local(
                FAISS_INDEX_PATH, 
                embeddings_model, 
                allow_dangerous_deserialization=True
            )
            
            # Get index statistics
            index_stats = vector_store.index.ntotal
            print(f"✅ Successfully loaded existing index")
            print(f"📊 Vectors in database: {index_stats:,}")
            
            # Test similarity search
            print("\n🔍 Testing similarity search...")
            test_query = "visa requirements"
            results = vector_store.similarity_search(test_query, k=3)
            
            print(f"✅ Similarity search test successful")
            print(f"📊 Retrieved {len(results)} similar documents")
            
            for i, doc in enumerate(results[:2], 1):
                preview = doc.page_content[:80].replace('\n', ' ')
                print(f"  Result {i}: {preview}...")
            
        else:
            print("🆕 Creating new FAISS index...")
            print(f"📊 Processing {len(document_chunks)} chunks...")
            
            # Create progress indicator
            batch_size = 50
            total_batches = (len(document_chunks) + batch_size - 1) // batch_size
            
            print(f"⚙️ Processing in batches of {batch_size}")
            print(f"⚙️ Total batches: {total_batches}")
            
            # Process in batches to show progress
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, len(document_chunks))
                
                print(f"  🔄 Processing batch {batch_num + 1}/{total_batches} " +
                      f"(chunks {start_idx + 1}-{end_idx})...")
                
                if batch_num == 0:
                    # Create initial vector store
                    batch_chunks = document_chunks[start_idx:end_idx]
                    vector_store = FAISS.from_texts(batch_chunks, embedding=embeddings_model)
                else:
                    # Add to existing vector store
                    batch_chunks = document_chunks[start_idx:end_idx]
                    batch_vector_store = FAISS.from_texts(batch_chunks, embedding=embeddings_model)
                    vector_store.merge_from(batch_vector_store)
            
            print("💾 Saving FAISS index to disk...")
            vector_store.save_local(FAISS_INDEX_PATH)
            print("✅ FAISS index created and saved successfully")
            
            # Final statistics
            final_count = vector_store.index.ntotal
            print(f"📊 Final vector count: {final_count:,}")
            
            # Test the new index
            print("\n🧪 Testing new index...")
            test_results = vector_store.similarity_search("UK visa", k=2)
            print(f"✅ Index test successful - retrieved {len(test_results)} results")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in FAISS operations: {str(e)}")
        return False

def print_rag_setup():
    """Detailed RAG (Retrieval-Augmented Generation) setup"""
    global conversation_chain
    
    print("\n🤖 RAG SYSTEM SETUP")
    print("-" * 50)
    
    try:
        print("🔄 Initializing ChatGoogleGenerativeAI...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=GEMINI_API_KEY, 
            temperature=0.7
        )
        print("✅ Language model initialized")
        print(f"🔧 Model: gemini-1.5-flash")
        print(f"🔧 Temperature: 0.7 (balanced creativity/accuracy)")
        print(f"🔧 Max Tokens: ~1M (context window)")
        
        print("\n📝 Setting up prompt template...")
        prompt_template = """
        You are a UK Visa Assistant, a professional resource for UK visa applications and processes. 
        You provide direct, helpful answers about UK visas without referencing source materials or limitations.

        Guidelines:
        - Answer questions directly about the topic without mentioning "provided text" or "context"
        - Use HTML formatting: <strong>headings</strong> for emphasis, <br> for line breaks
        - Create bullet points using • symbol followed by space
        - Keep responses concise and helpful (2-3 paragraphs maximum)
        - If specific information isn't available, provide accurate general UK visa guidance
        - Be professional but friendly in tone
        - Include actionable next steps when relevant
        - For complex cases, suggest contacting official UK visa services
        - Never start responses with phrases like "The provided text" or "Based on the context"
        - Always respond as if you have comprehensive knowledge of UK visa processes
        - At the end of each response, add a relevant reference link to the official UK government website (https://www.gov.uk), based on the topic
        - When available, reference the exact page or section from which the information is drawn
        
        Format example:
        <strong>Student Visa Requirements</strong><br><br>
        To apply for a UK student visa, you need:<br>
        • Confirmation of Acceptance for Studies (CAS)<br>
        • Proof of English language proficiency<br>
        • Financial evidence showing you can support yourself<br><br>
        The application process typically takes 3-8 weeks to process.

        
        Context: {context}
        Question: {question}

        Answer:
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        print("✅ Prompt template configured")
        
        print("\n🔗 Setting up retrieval chain...")
        print("⚙️ Chain Type: RetrievalQA with 'stuff' strategy")
        print("⚙️ Retriever: FAISS similarity search")
        print("⚙️ Return Source Documents: False (clean responses)")
        
        conversation_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt}
        )
        
        print("✅ RAG chain setup complete")
        
        # Test the RAG system
        print("\n🧪 Testing RAG system...")
        test_query = "What documents do I need for a UK visa?"
        print(f"Test Query: '{test_query}'")
        
        print("🔍 Step 1: Retrieving relevant documents...")
        # Manual retrieval test
        retrieved_docs = vector_store.similarity_search(test_query, k=3)
        print(f"✅ Retrieved {len(retrieved_docs)} relevant chunks")
        
        print("🤖 Step 2: Generating AI response...")
        test_response = conversation_chain({'query': test_query})
        
        response_preview = test_response['result'][:100].replace('\n', ' ')
        print(f"✅ AI response generated successfully")
        print(f"📝 Response preview: {response_preview}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in RAG setup: {str(e)}")
        return False

def print_system_summary():
    """Print final system summary"""
    print("\n📊 SYSTEM INITIALIZATION SUMMARY")
    print("-" * 50)
    
    status_items = [
        ("PDF Processing", pdf_processed),
        ("Document Chunking", len(document_chunks) > 0),
        ("Embeddings Model", embeddings_model is not None),
        ("FAISS Vector Store", vector_store is not None),
        ("RAG Chain", conversation_chain is not None),
        ("API Key", GEMINI_API_KEY is not None)
    ]
    
    for item, status in status_items:
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {item}")
    
    if all(status for _, status in status_items):
        print("\n🎉 ALL SYSTEMS OPERATIONAL - READY TO SERVE!")
    else:
        print("\n⚠️ SOME SYSTEMS NOT OPERATIONAL - RUNNING IN FALLBACK MODE")
    
    print(f"\n🗄️ Database Stats:")
    if vector_store:
        print(f"  📊 Total vectors: {vector_store.index.ntotal:,}")
        print(f"  📊 Embedding dimension: {embedding_stats.get('dimension', 'Unknown')}")
        print(f"  📊 Total chunks processed: {len(document_chunks)}")
    
    print(f"\n⚙️ Performance Info:")
    print(f"  🖥️ Flask host: 0.0.0.0:5000")
    print(f"  🔧 Debug mode: {app.debug}")
    print(f"  📁 Working directory: {os.getcwd()}")
    
    print("\n" + "="*80)

def get_pdf_content(pdf_path):
    """Extract content from PDF file"""
    try:
        pdf_reader = PdfReader(pdf_path)
        content = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                content += extracted
        if not content:
            raise ValueError("No content extracted from the document.")
        return content
    except Exception as e:
        raise ValueError(f"Error reading document: {str(e)}")

def get_content_chunks(content):
    """Split content into chunks for processing"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(content)
    return chunks

def get_vector_store(chunks):
    """Create and save FAISS vector store"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=GEMINI_API_KEY
        )
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        return True
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        return False

def get_conversation_chain():
    """Initialize the conversation chain with Gemini AI"""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=GEMINI_API_KEY, 
            temperature=0.7
        )
        
        prompt_template = """
        You are a UK Visa Assistant, a professional resource for UK visa applications and processes. 
        You provide direct, helpful answers about UK visas without referencing source materials or limitations.
        
        Guidelines:
        - Answer questions directly about the topic without mentioning "provided text" or "context"
        - Use HTML formatting: <strong>headings</strong> for emphasis, <br> for line breaks
        - Create bullet points using • symbol followed by space
        - Keep responses concise and helpful (2-3 paragraphs maximum)
        - If specific information isn't available, provide accurate general UK visa guidance
        - Be professional but friendly in tone
        - Include actionable next steps when relevant
        - For complex cases, suggest contacting official UK visa services
        - Never start responses with phrases like "The provided text" or "Based on the context"
        - Always respond as if you have comprehensive knowledge of UK visa processes
        - At the end of each response, add a relevant reference link to the official UK government website (https://www.gov.uk), based on the topic
        - When available, reference the exact page or section from which the information is drawn
        
        Format example:
        <strong>Student Visa Requirements</strong><br><br>
        To apply for a UK student visa, you need:<br>
        • Confirmation of Acceptance for Studies (CAS)<br>
        • Proof of English language proficiency<br>
        • Financial evidence showing you can support yourself<br><br>
        The application process typically takes 3-8 weeks to process.


        
        Context: {context}
        Question: {question}

        Answer:
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=GEMINI_API_KEY
        )
        
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        conversation_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return conversation_chain
    except Exception as e:
        print(f"Error initializing conversation chain: {str(e)}")
        return None

def initialize_pdf_processing():
    """Initialize PDF processing and vector store creation with detailed logging"""
    global pdf_processed, conversation_chain, vector_store, embeddings_model, document_chunks
    
    try:
        # Step 1: PDF Analysis
        content = print_pdf_analysis()
        if not content:
            return False
        
        # Step 2: Text Chunking
        if not print_chunking_process(content):
            return False
        
        # Step 3: Embedding Setup
        if not print_embedding_process():
            return False
        
        # Step 4: FAISS Operations
        if not print_faiss_operations():
            return False
        
        # Step 5: RAG Setup
        if not print_rag_setup():
            return False
        
        pdf_processed = True
        return True
        
    except Exception as e:
        print(f"❌ Critical error in PDF processing: {str(e)}")
        return False

def handle_ai_query(user_question):
    """Handle user query with AI processing and detailed logging"""
    global conversation_chain
    
    try:
        logger.info(f"🔍 Processing query: '{user_question}'")
        
        if not conversation_chain:
            logger.warning("⚠️ RAG chain not available")
            return {
                'text': 'I\'m currently setting up the knowledge base. Please try again in a moment, or ask me general questions about UK visas.',
                'type': 'text',
                'quick_options': [
                    {'text': '📋 Check Requirements', 'action': 'requirements'},
                    {'text': '🔍 Find Visa Type', 'action': 'visa_type'}
                ]
            }
        
        # Log retrieval step
        logger.info("📚 Step 1: Retrieving relevant documents...")
        start_time = time.time()
        
        response = conversation_chain({'query': user_question})
        ai_response = response['result']
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"✅ Query processed successfully in {processing_time:.2f} seconds")
        logger.info(f"📝 Response length: {len(ai_response)} characters")
        
        # Format response for better readability
        formatted_response = ai_response.replace('\n\n', '<br><br>').replace('\n', '<br>')
        
        return {
            'text': formatted_response,
            'type': 'html',
            'quick_options': [
                {'text': '❓ Ask Another Question', 'action': 'help'},
                {'text': '📞 Contact Support', 'action': 'advisor'}
            ]
        }
    except Exception as e:
        logger.error(f"❌ Error in AI query: {str(e)}")
        return {
            'text': 'I encountered an issue processing your question. Please try rephrasing or contact support for assistance.',
            'type': 'text',
            'quick_options': [
                {'text': '🔄 Try Again', 'action': 'help'},
                {'text': '👥 Human Support', 'action': 'advisor'}
            ]
        }

# Fallback knowledge base for when AI is not available
FALLBACK_KNOWLEDGE_BASE = {
    'tourist': {
        'name': 'Tourist Visa (Standard Visitor)',
        'requirements': [
            'Valid passport (6+ months validity)',
            'Proof of accommodation in the UK',
            'Return flight tickets',
            'Bank statements (last 3 months)',
            'Travel insurance',
            'Invitation letter (if visiting family/friends)'
        ],
        'timeline': '3 weeks processing time',
        'fee': '£100',
        'description': 'For tourism, visiting family/friends, or short business trips up to 6 months'
    },
    'student': {
        'name': 'Student Visa (Tier 4)',
        'requirements': [
            'Confirmation of Acceptance for Studies (CAS)',
            'Proof of English proficiency (IELTS/TOEFL)',
            'Financial evidence (tuition + living costs)',
            'TB test results (from approved clinics)',
            'Academic transcripts and certificates'
        ],
        'timeline': '3-8 weeks processing time',
        'fee': '£363',
        'description': 'For studying at UK educational institutions'
    },
    'work': {
        'name': 'Skilled Worker Visa',
        'requirements': [
            'Job offer from UK employer with sponsor licence',
            'Certificate of Sponsorship (CoS)',
            'Proof of English knowledge (B1 level)',
            'Salary requirements met (£25,600+ or going rate)'
        ],
        'timeline': '3-8 weeks processing time',
        'fee': '£719-£1,423',
        'description': 'For employment in the UK with a licensed sponsor'
    },
    'family': {
        'name': 'Family Visa (Partner/Spouse)',
        'requirements': [
            'Relationship evidence (marriage certificate, photos)',
            'Financial requirements (£18,600+ annual income)',
            'English language test (A1 level)',
            'Accommodation proof'
        ],
        'timeline': '2-12 weeks processing time',
        'fee': '£1,538',
        'description': 'For joining family members who are UK citizens or settled persons'
    }
}

def get_fallback_response(message_lower):
    """Generate fallback response using predefined knowledge base"""
    if any(word in message_lower for word in ['tourist', 'tourism', 'visit', 'holiday']):
        visa_info = FALLBACK_KNOWLEDGE_BASE['tourist']
        return format_visa_response(visa_info)
    elif any(word in message_lower for word in ['student', 'study', 'university', 'college']):
        visa_info = FALLBACK_KNOWLEDGE_BASE['student']
        return format_visa_response(visa_info)
    elif any(word in message_lower for word in ['work', 'job', 'employment', 'skilled']):
        visa_info = FALLBACK_KNOWLEDGE_BASE['work']
        return format_visa_response(visa_info)
    elif any(word in message_lower for word in ['family', 'spouse', 'partner', 'marriage']):
        visa_info = FALLBACK_KNOWLEDGE_BASE['family']
        return format_visa_response(visa_info)
    else:
        return {
            'text': 'I can help you with UK visa applications. What specific information do you need?',
            'type': 'text',
            'quick_options': [
                {'text': '🏖️ Tourist Visa', 'action': 'tourist'},
                {'text': '🎓 Student Visa', 'action': 'student'},
                {'text': '💼 Work Visa', 'action': 'work'},
                {'text': '👨‍👩‍👧‍👦 Family Visa', 'action': 'family'}
            ]
        }

def format_visa_response(visa_info):
    """Format visa information into response"""
    response_text = f"<strong>{visa_info['name']}</strong><br><br>"
    response_text += "📋 <strong>Requirements:</strong><br>"
    response_text += "<br>".join([f"• {req}" for req in visa_info['requirements']])
    response_text += f"<br><br>⏰ <strong>Processing:</strong> {visa_info['timeline']}"
    response_text += f"<br>💷 <strong>Fee:</strong> {visa_info['fee']}"
    response_text += f"<br><br>{visa_info['description']}"
    
    return {
        'text': response_text,
        'type': 'html',
        'quick_options': [
            {'text': '❓ Ask More Questions', 'action': 'help'},
            {'text': '📞 Contact Advisor', 'action': 'advisor'}
        ]
    }

def process_message(message):
    """Process user message and return appropriate response"""
    message_lower = message.lower().strip()
    
    # Handle greetings
    if any(greeting in message_lower for greeting in ['hello', 'hi', 'hey']):
        return {
            'text': 'Hello! 👋 Welcome to the UK Visa Assistant. I\'m here to help you with UK visa applications and requirements. What can I assist you with today?',
            'type': 'text',
            'quick_options': [
                {'text': '📋 Check Requirements', 'action': 'requirements'},
                {'text': '🔍 Find Visa Type', 'action': 'visa_type'},
                {'text': '⏰ Processing Times', 'action': 'timeline'},
                {'text': '💷 Visa Fees', 'action': 'fees'}
            ]
        }
    
    # Handle quick responses
    if any(word in message_lower for word in ['thank', 'thanks']):
        return {
            'text': 'You\'re welcome! If you have any more questions about UK visas, feel free to ask. I\'m here to help! 😊',
            'type': 'text'
        }
    
    # Try AI processing first
    if pdf_processed and conversation_chain:
        return handle_ai_query(message)
    else:
        # Use fallback knowledge base
        return get_fallback_response(message_lower)

@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages with detailed logging"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        logger.info(f"💬 New chat message: '{user_message}'")
        
        # Process the message
        start_time = time.time()
        response = process_message(user_message)
        processing_time = time.time() - start_time
        
        logger.info(f"⚡ Message processed in {processing_time:.3f} seconds")
        
        # Simulate typing delay
        time.sleep(1)
        
        return jsonify({
            'success': True,
            'response': response
        })
    
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/quick_action', methods=['POST'])
def quick_action():
    """Handle quick action buttons with logging"""
    try:
        data = request.get_json()
        action = data.get('action', '').strip()
        
        logger.info(f"🔘 Quick action triggered: '{action}'")
        
        # Map actions to appropriate queries
        action_queries = {
            'requirements': 'What are the general requirements for UK visa applications?',
            'visa_type': 'What types of UK visas are available?',
            'timeline': 'What are the processing times for UK visas?',
            'fees': 'What are the fees for UK visa applications?',
            'tourist': 'Tell me about tourist visa requirements for the UK',
            'student': 'What are the requirements for a UK student visa?',
            'work': 'What do I need for a UK work visa?',
            'family': 'What are the requirements for UK family visas?',
            'documents': 'What documents do I need for a UK visa application?',
            'advisor': 'How can I contact a human advisor for UK visa help?'
        }
        
        query = action_queries.get(action, f'Tell me about {action}')
        
        if action == 'advisor':
            response = {
                "text": "To speak with a human advisor about your UK visa application:<br><br>📞 <strong>UKVI Phone:</strong> 0300 790 6268<br>📧 <strong>Email:</strong> ApplyOnlineE-Support@homeoffice.gov.uk<br>📞 <strong>UKVCAS:</strong> +44 (0) 844 8920 232<br>📞 <strong>Immigration Enforcement:</strong> 0300 123 7000<br>📞 <strong>Passenger Support:</strong> 0800 876 6921 / 0203 337 0927<br>📧 <strong>Complaints:</strong> complaints@homeoffice.gov.uk<br>⚠️ <strong>Report Scams:</strong> 0300 123 2040<br>🚨 <strong>Emergency:</strong> Dial 999<br>🌐 <strong>Website:</strong> <a href='https://www.gov.uk/visas-immigration' target='_blank'>www.gov.uk/visas-immigration</a>",
                'type': 'html'
            }
        else:
            response = process_message(query)
        
        time.sleep(1)
        
        return jsonify({
            'success': True,
            'response': response
        })
    
    except Exception as e:
        logger.error(f"❌ Error in quick_action endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint with detailed system status"""
    system_info = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'pdf_processed': pdf_processed,
        'ai_available': conversation_chain is not None,
        'vector_store_loaded': vector_store is not None,
        'embeddings_ready': embeddings_model is not None,
        'total_chunks': len(document_chunks),
        'api_key_configured': GEMINI_API_KEY is not None
    }
    
    if vector_store:
        system_info['vector_count'] = vector_store.index.ntotal
    
    logger.info(f"🏥 Health check requested - Status: {system_info}")
    return jsonify(system_info)

@app.route('/system_stats')
def system_stats():
    """Detailed system statistics endpoint"""
    stats = {
        'initialization_time': datetime.now().isoformat(),
        'pdf_analysis': {
            'file_exists': os.path.exists(PDF_PATH),
            'file_size_mb': os.path.getsize(PDF_PATH) / (1024 * 1024) if os.path.exists(PDF_PATH) else 0,
            'chunks_created': len(document_chunks)
        },
        'embedding_stats': embedding_stats,
        'vector_database': {
            'index_exists': os.path.exists(FAISS_INDEX_PATH),
            'vector_count': vector_store.index.ntotal if vector_store else 0,
            'embedding_dimension': embedding_stats.get('dimension', 0)
        },
        'rag_system': {
            'chain_ready': conversation_chain is not None,
            'model': 'gemini-1.5-flash',
            'temperature': 0.7
        },
        'flask_config': {
            'debug_mode': app.debug,
            'host': '0.0.0.0',
            'port': 5000
        }
    }
    
    return jsonify(stats)

if __name__ == '__main__':
    # Print detailed startup banner
    print_banner()
    
    if not GEMINI_API_KEY:
        print("\n❌ ERROR: GOOGLE_API_KEY not found in environment variables!")
        print("🔧 Please ensure you have a .env file with your Google API key.")
        print("⚠️ Running in fallback mode with limited functionality.")
    else:
        print("✅ Google API key configured successfully")
        print("\n🔄 Starting comprehensive system initialization...")
        
        # Initialize with detailed logging
        success = initialize_pdf_processing()
        
        if success:
            pdf_processed = True
            print("\n🎉 INITIALIZATION SUCCESSFUL!")
        else:
            print("\n⚠️ INITIALIZATION FAILED - Running in fallback mode")
    
    # Print final system summary
    print_system_summary()
    
    # Additional startup information
    print(f"\n🌐 FLASK SERVER STARTING...")
    print(f"🔗 Local access: http://127.0.0.1:5000")
    print(f"🔗 Network access: http://192.168.1.8:5000")
    print(f"📊 Health check: http://127.0.0.1:5000/health")
    print(f"📈 System stats: http://127.0.0.1:5000/system_stats")
    print(f"📝 Log file: visa_assistant.log")
    print("\n💡 Use Ctrl+C to stop the server")
    print("="*80)
    
    app.run(debug=True, host='0.0.0.0', port=5000)