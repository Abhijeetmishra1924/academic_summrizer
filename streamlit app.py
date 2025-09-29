# app.py
import streamlit as st
import fitz  # PyMuPDF
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# =====================
# CONFIGURATION
# =====================
st.set_page_config(page_title="Academic Paper Summarizer", page_icon="📑", layout="wide")

# 🔑 Load Groq API key from Streamlit secrets
GROQ_API_KEY = st.secrets["api_key"]

# =====================
# HELPER FUNCTIONS
# =====================
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF using PyMuPDF"""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def generate_summary(text, focus):
    """Generate a summary using Groq LLM with chunking"""
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )

    template = """
    You are an AI academic assistant. Summarize the following research paper text.
    Focus specifically on: {focus}.
    Extract key insights clearly and concisely. 
    Avoid generic statements; emphasize methodology, results, and limitations if available.

    Research Paper Text:
    {paper_text}
    """

    prompt = PromptTemplate(input_variables=["focus", "paper_text"], template=template)

    # Split text into chunks to avoid token overflow
    max_chars = 4000
    chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

    summary = ""
    for chunk in chunks:
        final_prompt = prompt.format(focus=focus, paper_text=chunk)
        response = llm.invoke(final_prompt)
        summary += response.content + "\n\n"

    return summary


def answer_question(text, question):
    """Answer user question from the paper using Groq LLM with chunking"""
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )

    template = """
    You are an academic research assistant. Based only on the provided research paper text, 
    answer the following question in detail and cite relevant sections if possible.

    Question: {question}
    Research Paper Text: {paper_text}
    """

    prompt = PromptTemplate(input_variables=["question", "paper_text"], template=template)

    max_chars = 4000
    chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

    answer = ""
    for chunk in chunks:
        final_prompt = prompt.format(question=question, paper_text=chunk)
        response = llm.invoke(final_prompt)
        answer += response.content + "\n\n"

    return answer


# =====================
# STREAMLIT UI
# =====================
st.title("📑 Academic Paper Summarizer")
st.markdown("Upload a research paper PDF and get **smart summaries, Q&A, and insights** powered by Groq LLM.")

uploaded_file = st.file_uploader("📂 Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("🔍 Extracting text from PDF..."):
        paper_text = extract_text_from_pdf(uploaded_file)

    st.success("✅ PDF text extracted successfully!")

    # =====================
    # SUMMARY SECTION
    # =====================
    st.subheader("📌 Generate Summary")
    focus_option = st.selectbox(
        "🎯 Choose summary focus:",
        ["General Overview", "Methodology", "Results & Findings", "Key Takeaways", "Limitations & Future Work"]
    )

    if st.button("✨ Summarize"):
        with st.spinner("⚡ Generating summary..."):
            summary = generate_summary(paper_text, focus_option)
        st.subheader("📝 Summary")
        st.write(summary)

    # =====================
    # Q&A SECTION
    # =====================
    st.subheader("❓ Ask Questions About the Paper")
    user_question = st.text_input("Type your question (e.g., What dataset was used?)")
    if st.button("🔎 Get Answer") and user_question:
        with st.spinner("🤔 Thinking..."):
            answer = answer_question(paper_text, user_question)
        st.subheader("📖 Answer")
        st.write(answer)
