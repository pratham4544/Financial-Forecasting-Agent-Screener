import streamlit as st
import os
from research.raw import *
import json

# Page configuration
st.set_page_config(
    page_title="Financial Document Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'url' not in st.session_state:
    st.session_state.url = ''
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

DOWNLOAD_DIR = "pdf_downloads"

# Header
st.markdown('<div class="main-header">📊 Financial Document Analyzer</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("📁 Document Processing")
    if st.button("🗑️ Reset Download Folder"):
        with st.spinner("Resetting download folder..."):
            reset_download_folder()
            st.success("✅ Download folder reset successfully!")
    
    st.markdown("---")
    
    st.subheader("💾 Vector Database")
    if st.button("💾 Save Vector DB"):
        if st.session_state.vector_db:
            with st.spinner("Saving vector database..."):
                st.session_state.vector_db.save_local('faiss_index')
                st.success("✅ Vector DB saved to 'faiss_index'")
        else:
            st.warning("⚠️ No vector database to save!")
    
    if st.button("📂 Load Vector DB"):
        try:
            with st.spinner("Loading vector database..."):
                st.session_state.vector_db = FAISS.load_local(
                    'faiss_index',
                    embeddings=embeddings_model,
                    allow_dangerous_deserialization=True
                )
                st.session_state.processing_complete = True
                st.success("✅ Vector DB loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading vector DB: {str(e)}")
    
    st.markdown("---")
    
    st.subheader("ℹ️ About")
    st.info("""
    This app analyzes financial documents from Screener.in:
    
    1. Enter a company URL
    2. Process documents
    3. Ask questions about financials
    4. Get AI-powered insights
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["📥 Document Processing", "💬 Query Analysis", "📜 Chat History"])

with tab1:
    st.header("Document Processing Pipeline")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        url_input = st.text_input(
            "Enter Screener.in Company URL",
            value="https://www.screener.in/company/GROWW/consolidated/",
            placeholder="https://www.screener.in/company/COMPANY_NAME/consolidated/"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        process_btn = st.button("🚀 Process Documents", type="primary")
    
    if process_btn and url_input:
        st.session_state.url = url_input
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Reset folder
            status_text.text("🗑️ Resetting download folder...")
            reset_download_folder()
            progress_bar.progress(10)
            
            # Step 2: Download PDFs
            status_text.text("📥 Downloading PDFs from Screener.in...")
            run(url_input)
            progress_bar.progress(30)
            st.success("✅ PDFs downloaded successfully!")
            
            # Step 3: Delete old PDFs
            status_text.text("🧹 Removing old PDFs (>1 year)...")
            delete_result = delete_old_pdfs()
            progress_bar.progress(40)
            st.info(f"ℹ️ {delete_result}")
            
            # Step 4: Create chunks
            status_text.text("✂️ Creating document chunks...")
            chunks = create_chunks(DOWNLOAD_DIR)
            progress_bar.progress(60)
            st.success(f"✅ Created {len(chunks)} document chunks")
            
            # Step 5: Create PDF vector store
            status_text.text("🗄️ Building vector database from PDFs...")
            vector_db = create_pdf_vector_stores(chunks)
            progress_bar.progress(80)
            
            # Step 6: Add URL content
            status_text.text("🌐 Adding URL content to vector database...")
            vector_db = create_url_vector_store(url_input, vector_db)
            progress_bar.progress(95)
            
            # Step 7: Complete
            st.session_state.vector_db = vector_db
            st.session_state.processing_complete = True
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            
            st.markdown('<div class="success-box">🎉 All documents processed successfully! You can now query the database.</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            progress_bar.progress(0)
    
    # Display processing status
    if st.session_state.processing_complete:
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Status", "Ready ✅", delta="Vector DB Loaded")
        
        with col2:
            if st.session_state.url:
                st.metric("Current URL", "Loaded", delta=st.session_state.url.split('/')[-3] if st.session_state.url else "N/A")

with tab2:
    st.header("Ask Questions About Financial Documents")
    
    if not st.session_state.processing_complete:
        st.warning("⚠️ Please process documents first in the 'Document Processing' tab!")
    else:
        # Query input
        query = st.text_area(
            "Enter your financial analysis query:",
            placeholder="Example: What is the revenue growth trend? What are the key risks mentioned?",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_btn = st.button("🔍 Analyze", type="primary")
        
        if analyze_btn and query:
            with st.spinner("🤔 Analyzing documents..."):
                try:
                    response, extracted_chunks = user_query_answer(query, st.session_state.vector_db)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "query": query,
                        "response": response,
                        "chunks": extracted_chunks
                    })
                    
                    # Display response
                    st.markdown("### 📊 Analysis Results")
                    
                    # Parse and display JSON response
                    if isinstance(response, dict):
                        if 'reply' in response:
                            st.markdown('<div class="info-box">', unsafe_allow_html=True)
                            st.markdown(f"**💡 Answer:**\n\n{response['reply']}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        if 'guidance_caution' in response:
                            st.caption(f"⚠️ {response['guidance_caution']}")
                        
                        if 'follow_up_prompt' in response:
                            st.info(f"💭 {response['follow_up_prompt']}")
                        
                        # Show full JSON in expander
                        with st.expander("📄 View Full JSON Response"):
                            st.json(response)
                    else:
                        st.write(response)
                    
                    # Get current market price
                    st.markdown("---")
                    st.subheader("📈 Current Market Data")
                    
                    try:
                        cmp = current_market_price(st.session_state.url)
                        if cmp:
                            st.metric("Current Market Price", f"₹{cmp}", delta="Live Price")
                        else:
                            st.warning("⚠️ Could not fetch current market price")
                    except Exception as e:
                        st.error(f"❌ Error fetching market price: {str(e)}")
                    
                    # Show relevant chunks
                    with st.expander("📚 View Source Document Chunks"):
                        for i, (doc, score) in enumerate(extracted_chunks, 1):
                            st.markdown(f"**Chunk {i}** (Relevance Score: {score:.4f})")
                            st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                            st.markdown("---")
                    
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")

with tab3:
    st.header("Chat History")
    
    if not st.session_state.chat_history:
        st.info("💬 No chat history yet. Start asking questions in the 'Query Analysis' tab!")
    else:
        for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
            with st.expander(f"Query {len(st.session_state.chat_history) - i + 1}: {chat['query'][:100]}..."):
                st.markdown(f"**❓ Query:**\n{chat['query']}")
                st.markdown("---")
                st.markdown("**💡 Response:**")
                if isinstance(chat['response'], dict):
                    st.json(chat['response'])
                else:
                    st.write(chat['response'])
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Built with Streamlit | Powered by LangChain & GROQ</div>",
    unsafe_allow_html=True
)