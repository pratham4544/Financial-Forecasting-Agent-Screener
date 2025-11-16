import streamlit as st
import research.raw_code as engine  # <-- your raw code, NOT modified

st.set_page_config(page_title="Financial Forecasting Agent", layout="wide")

# ---------------------------------------
# Initialize session state
# ---------------------------------------
if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False

if "url" not in st.session_state:
    st.session_state.url = None


# ---------------------------------------
# UI Title
# ---------------------------------------
st.title("📊 Financial Forecasting Agent")
st.write("Analyze quarterly reports & earnings transcripts using your RAG pipeline.")


# ---------------------------------------
# Step 1 — User provides URL
# ---------------------------------------
st.subheader("🔗 Step 1: Enter Screener URL")

url = st.text_input("Enter company Screener URL:", placeholder="https://www.screener.in/company/TCS/consolidated/")

if st.button("Load & Build Vector Store"):
    if not url:
        st.error("Please enter a valid URL.")
    else:
        st.session_state.url = url
        st.info("📥 Downloading PDFs... This may take 10–60 seconds.")
        msg = engine.download_pdfs(url)

        st.info("🔄 Creating vector store...")
        vs = engine.create_vector_store(url)

        if vs:
            st.session_state.vector_ready = True
            st.success("✅ Vector store created successfully!")
        else:
            st.error("❌ Failed to create vector store.")

        st.write(msg)


# ---------------------------------------
# Step 2 — Ask questions (only after vector store exists)
# ---------------------------------------
st.subheader("💬 Step 2: Ask Questions")

if not st.session_state.vector_ready:
    st.warning("Load a Screener URL first to enable Q&A.")
else:
    question = st.text_area("Your question:", placeholder="Analyze the financial reports for the last 2 quarters...")

    if st.button("Ask"):
        with st.spinner("🔍 Retrieving info & generating insights..."):
            try:
                answer, chunks = engine.user_query_answer(question)
                st.success("Response generated!")
                
                st.subheader("📘 Model Response (JSON)")
                st.json(answer)

                st.subheader("📚 Chunks Used")
                st.write(f"{len(chunks)} chunks retrieved.")
                st.write(chunks)

            except Exception as e:
                st.err
