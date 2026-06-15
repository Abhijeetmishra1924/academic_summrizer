import streamlit as st
import fitz  # PyMuPDF
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import re
import json
from datetime import datetime
import io

# =====================
# PAGE CONFIG 
# =====================
st.set_page_config(
    page_title="PaperLens Pro · Academic Summarizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================
# CUSTOM CSS
# =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

/* Root theme */
:root {
    --bg-primary: #0D1117;
    --bg-card: #161B22;
    --bg-hover: #1C2128;
    --accent: #58A6FF;
    --accent-2: #7EE787;
    --accent-3: #F78166;
    --text-primary: #E6EDF3;
    --text-secondary: #8B949E;
    --text-muted: #484F58;
    --border: #30363D;
    --border-light: #21262D;
    --glow: rgba(88, 166, 255, 0.15);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem !important; max-width: 1300px; }

/* ---- HEADER ---- */
.paperlens-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 28px 0 8px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 28px;
}
.paperlens-logo {
    font-size: 2.6rem;
}
.paperlens-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.1;
    margin: 0;
}
.paperlens-subtitle {
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin: 4px 0 0 0;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.badge-pro {
    background: linear-gradient(135deg, #58A6FF, #7EE787);
    color: #0D1117;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 12px;
    letter-spacing: 0.06em;
    vertical-align: middle;
    margin-left: 8px;
}

/* ---- CARDS ---- */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
}
.card-accent {
    border-left: 3px solid var(--accent);
}
.card-green {
    border-left: 3px solid var(--accent-2);
}
.card-red {
    border-left: 3px solid var(--accent-3);
}

/* ---- METRIC ROW ---- */
.metric-row {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.metric-box {
    flex: 1;
    min-width: 120px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.metric-label {
    font-size: 0.72rem;
    color: var(--text-secondary);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ---- SECTION TITLES ---- */
.section-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-light);
}

/* ---- RESULT BOX ---- */
.result-box {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 18px 20px;
    font-size: 0.92rem;
    line-height: 1.75;
    color: var(--text-primary);
    white-space: pre-wrap;
}

/* ---- TAG PILLS ---- */
.tag-pill {
    display: inline-block;
    background: rgba(88, 166, 255, 0.12);
    color: var(--accent);
    border: 1px solid rgba(88, 166, 255, 0.25);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 3px 3px 3px 0;
}

/* ---- SIDEBAR ---- */
section[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem !important;
}

/* ---- BUTTONS ---- */
.stButton > button {
    background: var(--accent) !important;
    color: #0D1117 !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.4rem !important;
    font-size: 0.88rem !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}

/* ---- INPUTS ---- */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    background: var(--bg-primary) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}

/* ---- TABS ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--bg-card);
    border-radius: 10px;
    padding: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #0D1117 !important;
}

/* ---- EXPANDER ---- */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}

/* ---- FILE UPLOADER ---- */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
}

/* ---- HIGHLIGHT BLOCK ---- */
.highlight-block {
    background: rgba(88, 166, 255, 0.06);
    border: 1px solid rgba(88, 166, 255, 0.18);
    border-radius: 8px;
    padding: 14px 16px;
    margin: 10px 0;
    font-size: 0.88rem;
    line-height: 1.65;
}

/* ---- SPINNER ---- */
.stSpinner > div > div {
    border-top-color: var(--accent) !important;
}

/* ---- DIVIDER ---- */
.custom-divider {
    height: 1px;
    background: var(--border);
    margin: 20px 0;
}

/* ---- STATUS BADGES ---- */
.status-ok {
    display: inline-block;
    background: rgba(126, 231, 135, 0.12);
    color: var(--accent-2);
    border: 1px solid rgba(126, 231, 135, 0.25);
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* ---- CHAT BUBBLES ---- */
.chat-user {
    background: rgba(88, 166, 255, 0.1);
    border: 1px solid rgba(88, 166, 255, 0.2);
    border-radius: 12px 12px 4px 12px;
    padding: 10px 14px;
    margin: 8px 0 8px 40px;
    font-size: 0.88rem;
}
.chat-ai {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px 12px 12px 4px;
    padding: 10px 14px;
    margin: 8px 40px 8px 0;
    font-size: 0.88rem;
    line-height: 1.7;
}
.chat-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.chat-label-user { color: var(--accent); }
.chat-label-ai { color: var(--accent-2); }
</style>
""", unsafe_allow_html=True)

# =====================
# INIT SESSION STATE
# =====================
if "paper_text" not in st.session_state:
    st.session_state.paper_text = ""
if "paper_name" not in st.session_state:
    st.session_state.paper_name = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summaries" not in st.session_state:
    st.session_state.summaries = {}
if "key_entities" not in st.session_state:
    st.session_state.key_entities = ""

# =====================
# SECRETS & LLM
# =====================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

def get_llm(temperature=0.3):
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=temperature)

# =====================
# HELPER FUNCTIONS
# =====================
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text, len(doc)

def count_words(text):
    return len(re.findall(r'\w+', text))

def truncate(text, max_chars=7000):
    return text[:max_chars]

# ---- GENERATE SUMMARY ----
def generate_summary(text, focus, length="Detailed"):
    llm = get_llm()
    length_map = {
        "Brief": "in 150–200 words",
        "Standard": "in 300–400 words",
        "Detailed": "in 500–700 words with structured sections",
    }
    length_instr = length_map.get(length, "in 400 words")
    template = """You are a senior academic researcher with expertise in summarizing complex papers.
Summarize the following research paper {length_instr}.
Focus specifically on: {focus}.

Structure your response clearly with:
- **Core Contribution**: What is novel about this work?
- **Methodology**: How did they do it?
- **Key Results**: What were the main findings?
- **Limitations**: What are the weaknesses or gaps?
- **So What**: Why does this matter to the field?

Be precise. Avoid filler. Use specific numbers, names, and claims from the paper where available.

Research Paper Text:
{paper_text}
"""
    prompt = PromptTemplate(input_variables=["focus", "paper_text", "length_instr"], template=template)
    final_prompt = prompt.format(focus=focus, paper_text=truncate(text), length_instr=length_instr)
    return get_llm().invoke(final_prompt).content

# ---- EXTRACT KEY ENTITIES ----
def extract_entities(text):
    template = """From the research paper below, extract and return a JSON object with these keys:
- "authors": list of author names (max 5)
- "year": publication year (string or null)
- "keywords": list of 6–10 domain keywords
- "dataset": list of datasets mentioned
- "models": list of models, algorithms, or methods used
- "metrics": list of evaluation metrics used
- "institution": list of institutions or universities

Return ONLY valid JSON, no markdown, no explanation.

Paper text:
{paper_text}
"""
    prompt = PromptTemplate(input_variables=["paper_text"], template=template)
    final_prompt = prompt.format(paper_text=truncate(text, 3000))
    response = get_llm(temperature=0.1).invoke(final_prompt).content
    try:
        clean = response.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception:
        return {}

# ---- Q&A ----
def answer_question(text, question, history):
    history_str = "\n".join(
        [f"User: {h['q']}\nAssistant: {h['a']}" for h in history[-4:]]
    )
    template = """You are a precise academic research assistant. You only answer based on the provided paper.
If the answer isn't in the paper, say so honestly.

Previous conversation:
{history}

Question: {question}

Research Paper:
{paper_text}

Provide a thorough answer with specific references to sections, figures, or results in the paper where applicable.
"""
    prompt = PromptTemplate(input_variables=["question", "paper_text", "history"], template=template)
    final_prompt = prompt.format(question=question, paper_text=truncate(text), history=history_str)
    return get_llm().invoke(final_prompt).content

# ---- CRITICAL ANALYSIS ----
def critical_analysis(text):
    template = """You are a rigorous peer reviewer for a top-tier journal.

Provide a critical analysis of the paper below with these sections:

**Strengths** (bullet list, 3–5 points)
**Weaknesses** (bullet list, 3–5 points)
**Methodological Concerns** (2–3 specific issues)
**Reproducibility Assessment** (brief paragraph)
**Suggested Future Work** (3 concrete directions)
**Overall Verdict** (1–2 sentences, honest assessment)

Paper text:
{paper_text}
"""
    prompt = PromptTemplate(input_variables=["paper_text"], template=template)
    return get_llm().invoke(prompt.format(paper_text=truncate(text))).content

# ---- COMPARE TWO PAPERS ----
def compare_papers(text1, text2, name1, name2):
    template = """You are a research analyst. Compare these two research papers across these dimensions:

1. **Research Goals** – What each paper aims to solve
2. **Methodology** – Approaches and techniques used
3. **Datasets & Evaluation** – What was tested, metrics used
4. **Key Results** – Main performance or findings
5. **Novelty** – What each contributes uniquely
6. **Limitations** – Weaknesses of each
7. **Verdict** – Which paper is stronger and why

Paper A ({name1}):
{text1}

Paper B ({name2}):
{text2}

Be specific and avoid generic comparisons.
"""
    prompt = PromptTemplate(input_variables=["text1", "text2", "name1", "name2"], template=template)
    final = prompt.format(text1=truncate(text1, 3500), text2=truncate(text2, 3500), name1=name1, name2=name2)
    return get_llm().invoke(final).content

# ---- GENERATE PLAIN LANGUAGE EXPLAINER ----
def explain_simply(text):
    template = """Explain this research paper as if the reader is a smart high school student with no domain expertise.
Avoid jargon. Use analogies. Keep it engaging and clear. Around 250 words.

Paper:
{paper_text}
"""
    return get_llm().invoke(
        PromptTemplate(input_variables=["paper_text"], template=template).format(paper_text=truncate(text, 4000))
    ).content

# ---- GENERATE TWEET THREAD ----
def generate_tweet_thread(text):
    template = """Create a Twitter/X thread of 5 tweets summarizing this research paper.
Format: 
Tweet 1/5: [hook]
Tweet 2/5: [problem being solved]
Tweet 3/5: [method/approach]
Tweet 4/5: [key result with numbers if possible]
Tweet 5/5: [why it matters + conclusion]

Keep each tweet under 260 characters. Be punchy and accurate.

Paper:
{paper_text}
"""
    return get_llm().invoke(
        PromptTemplate(input_variables=["paper_text"], template=template).format(paper_text=truncate(text, 4000))
    ).content

# =====================
# SIDEBAR
# =====================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; margin-bottom: 20px;'>
        <div style='font-size:2.2rem;'>🔬</div>
        <div style='font-family: Playfair Display, serif; font-size:1.1rem; font-weight:700; color:#E6EDF3;'>PaperLens <span style='background:linear-gradient(135deg,#58A6FF,#7EE787);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>Pro</span></div>
        <div style='font-size:0.7rem; color:#8B949E; margin-top:2px;'>ACADEMIC PAPER INTELLIGENCE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-title">📂 Upload Paper(s)</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Primary PDF", type=["pdf"], label_visibility="collapsed")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">⚙️ Summary Settings</p>', unsafe_allow_html=True)

    focus_option = st.selectbox(
        "Summary Focus",
        ["General Overview", "Methodology", "Results & Findings", "Key Takeaways", "Limitations & Future Work", "Technical Deep-Dive"]
    )
    summary_length = st.radio("Summary Length", ["Brief", "Standard", "Detailed"], index=2, horizontal=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🔄 Compare Mode</p>', unsafe_allow_html=True)
    uploaded_file_2 = st.file_uploader("Second PDF (optional)", type=["pdf"], label_visibility="collapsed")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem; color:#484F58; line-height:1.6;'>
    Powered by <b style='color:#8B949E;'>Groq LLaMA 3.3 70B</b><br>
    PaperLens Pro v2.0
    </div>
    """, unsafe_allow_html=True)

# =====================
# MAIN HEADER
# =====================
st.markdown("""
<div class="paperlens-header">
    <div class="paperlens-logo">🔬</div>
    <div>
        <div class="paperlens-title">PaperLens <span class="badge-pro">PRO</span></div>
        <div class="paperlens-subtitle">Academic Paper Intelligence · Summarize · Analyze · Explore</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================
# PDF PROCESSING
# =====================
if uploaded_file:
    if st.session_state.paper_name != uploaded_file.name:
        with st.spinner("Extracting and indexing paper..."):
            text, num_pages = extract_text_from_pdf(uploaded_file)
            st.session_state.paper_text = text
            st.session_state.paper_name = uploaded_file.name
            st.session_state.chat_history = []
            st.session_state.summaries = {}
            st.session_state.key_entities = ""

    paper_text = st.session_state.paper_text
    word_count = count_words(paper_text)

    # ---- PAPER STATS ----
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-box"><div class="metric-value">{word_count:,}</div><div class="metric-label">Words</div></div>""", unsafe_allow_html=True)
    with col2:
        est_pages = max(1, word_count // 350)
        st.markdown(f"""<div class="metric-box"><div class="metric-value">{est_pages}</div><div class="metric-label">Est. Pages</div></div>""", unsafe_allow_html=True)
    with col3:
        read_time = max(1, word_count // 200)
        st.markdown(f"""<div class="metric-box"><div class="metric-value">{read_time} min</div><div class="metric-label">Read Time</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-box"><div class="metric-value"><span class="status-ok">Ready</span></div><div class="metric-label">Status</div></div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card" style="margin-top:4px;">
        <span style="font-size:0.78rem; color:#8B949E;">📄 Loaded:</span>
        <span style="font-size:0.88rem; font-weight:600; color:#E6EDF3; margin-left:8px;">{uploaded_file.name}</span>
    </div>
    """, unsafe_allow_html=True)

    # =====================
    # TABS
    # =====================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📝 Summary", "🔍 Deep Analysis", "💬 Research Chat", "📊 Paper Profile", "✍️ Export & Share", "🆚 Compare"
    ])

    # ==========
    # TAB 1: SUMMARY
    # ==========
    with tab1:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown('<p class="section-title">Smart Summary</p>', unsafe_allow_html=True)
        with col_b:
            summarize_btn = st.button("✨ Generate Summary", use_container_width=True)

        cache_key = f"{focus_option}_{summary_length}"

        if summarize_btn:
            with st.spinner(f"Generating {summary_length.lower()} summary focused on {focus_option}..."):
                summary = generate_summary(paper_text, focus_option, summary_length)
                st.session_state.summaries[cache_key] = summary

        if cache_key in st.session_state.summaries:
            summary = st.session_state.summaries[cache_key]
            st.markdown(f"""
            <div class="card card-accent">
                <div class="section-title" style="margin-bottom:14px;">
                    Focus: {focus_option} · {summary_length}
                </div>
                <div class="result-box">{summary}</div>
            </div>
            """, unsafe_allow_html=True)

            # Download summary
            st.download_button(
                "⬇️ Download Summary (.txt)",
                data=f"PaperLens Pro Summary\n{'='*40}\nPaper: {uploaded_file.name}\nFocus: {focus_option}\nLength: {summary_length}\nDate: {datetime.now().strftime('%Y-%m-%d')}\n\n{summary}",
                file_name=f"summary_{uploaded_file.name.replace('.pdf','')}.txt",
                mime="text/plain"
            )
        else:
            st.markdown("""
            <div class="card" style="text-align:center; padding:40px; color:#484F58;">
                <div style="font-size:2rem;">📝</div>
                <div style="margin-top:10px; font-size:0.9rem;">Choose your focus and length in the sidebar, then click Generate Summary</div>
            </div>
            """, unsafe_allow_html=True)

    # ==========
    # TAB 2: DEEP ANALYSIS
    # ==========
    with tab2:
        st.markdown('<p class="section-title">Critical Peer Review Analysis</p>', unsafe_allow_html=True)

        col_x, col_y = st.columns(2)
        with col_x:
            analyze_btn = st.button("🔍 Run Critical Analysis", use_container_width=True)
        with col_y:
            explain_btn = st.button("🧠 Plain Language Explainer", use_container_width=True)

        if analyze_btn:
            with st.spinner("Performing peer-review style analysis..."):
                analysis = critical_analysis(paper_text)
            st.markdown(f"""
            <div class="card card-red">
                <div class="section-title">Critical Analysis</div>
                <div class="result-box">{analysis}</div>
            </div>
            """, unsafe_allow_html=True)

        if explain_btn:
            with st.spinner("Translating to plain language..."):
                plain = explain_simply(paper_text)
            st.markdown(f"""
            <div class="card card-green">
                <div class="section-title">Plain Language Explainer</div>
                <div class="result-box">{plain}</div>
            </div>
            """, unsafe_allow_html=True)

    # ==========
    # TAB 3: RESEARCH CHAT
    # ==========
    with tab3:
        st.markdown('<p class="section-title">Chat With Your Paper</p>', unsafe_allow_html=True)

        # Display history
        for item in st.session_state.chat_history:
            st.markdown(f"""
            <div class="chat-user"><div class="chat-label chat-label-user">You</div>{item['q']}</div>
            <div class="chat-ai"><div class="chat-label chat-label-ai">PaperLens</div>{item['a']}</div>
            """, unsafe_allow_html=True)

        # Suggested questions
        if not st.session_state.chat_history:
            st.markdown("""
            <div style='font-size:0.78rem; color:#8B949E; margin-bottom:10px;'>💡 Try asking:</div>
            """, unsafe_allow_html=True)
            suggestions = [
                "What is the main research problem?",
                "What dataset and evaluation metrics were used?",
                "What are the key limitations of this work?",
                "How does this compare to prior work?",
            ]
            cols = st.columns(2)
            for i, s in enumerate(suggestions):
                with cols[i % 2]:
                    if st.button(s, key=f"sug_{i}", use_container_width=True):
                        with st.spinner("Researching answer..."):
                            ans = answer_question(paper_text, s, st.session_state.chat_history)
                        st.session_state.chat_history.append({"q": s, "a": ans})
                        st.rerun()

        user_q = st.text_input("Ask anything about this paper...", placeholder="e.g. What baseline models were compared?", key="chat_input")
        col_ask, col_clear = st.columns([3, 1])
        with col_ask:
            ask_btn = st.button("→ Ask", use_container_width=True)
        with col_clear:
            if st.button("🗑 Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        if ask_btn and user_q.strip():
            with st.spinner("Analyzing paper to answer..."):
                answer = answer_question(paper_text, user_q, st.session_state.chat_history)
            st.session_state.chat_history.append({"q": user_q, "a": answer})
            st.rerun()

    # ==========
    # TAB 4: PAPER PROFILE
    # ==========
    with tab4:
        st.markdown('<p class="section-title">Extracted Paper Profile</p>', unsafe_allow_html=True)

        extract_btn = st.button("📊 Extract Paper Metadata", use_container_width=False)

        if extract_btn or st.session_state.key_entities:
            if not st.session_state.key_entities:
                with st.spinner("Extracting entities, keywords, and metadata..."):
                    entities = extract_entities(paper_text)
                    st.session_state.key_entities = entities
            else:
                entities = st.session_state.key_entities

            if entities:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("""<div class="card"><div class="section-title">Authorship</div>""", unsafe_allow_html=True)
                    for author in entities.get("authors", []):
                        st.markdown(f"👤 **{author}**")
                    if entities.get("institution"):
                        st.markdown("**Institutions:**")
                        for inst in entities.get("institution", []):
                            st.markdown(f"🏛️ {inst}")
                    if entities.get("year"):
                        st.markdown(f"📅 **Year:** {entities.get('year')}")
                    st.markdown("</div>", unsafe_allow_html=True)

                with col2:
                    st.markdown("""<div class="card"><div class="section-title">Methods & Evaluation</div>""", unsafe_allow_html=True)
                    if entities.get("models"):
                        st.markdown("**Models / Methods:**")
                        for m in entities.get("models", []):
                            st.markdown(f"`{m}`")
                    if entities.get("metrics"):
                        st.markdown("**Metrics:**")
                        for me in entities.get("metrics", []):
                            st.markdown(f"📐 {me}")
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("""<div class="card"><div class="section-title">Keywords & Datasets</div>""", unsafe_allow_html=True)
                kw_html = "".join([f'<span class="tag-pill">{k}</span>' for k in entities.get("keywords", [])])
                st.markdown(kw_html, unsafe_allow_html=True)
                if entities.get("dataset"):
                    st.markdown("**Datasets used:**")
                    for d in entities.get("dataset", []):
                        st.markdown(f"🗃️ {d}")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Could not extract structured metadata. Try with a clearer PDF.")

        with st.expander("📄 View Raw Extracted Text (first 3000 chars)"):
            st.code(paper_text[:3000], language=None)

    # ==========
    # TAB 5: EXPORT & SHARE
    # ==========
    with tab5:
        st.markdown('<p class="section-title">Export & Share Your Analysis</p>', unsafe_allow_html=True)

        col_e1, col_e2 = st.columns(2)

        with col_e1:
            st.markdown('<div class="card"><div class="section-title">🐦 Twitter/X Thread</div>', unsafe_allow_html=True)
            tweet_btn = st.button("Generate Tweet Thread", use_container_width=True)
            if tweet_btn:
                with st.spinner("Writing viral thread..."):
                    thread = generate_tweet_thread(paper_text)
                st.markdown(f'<div class="result-box">{thread}</div>', unsafe_allow_html=True)
                st.download_button("⬇️ Download Thread", data=thread, file_name="thread.txt", mime="text/plain")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_e2:
            st.markdown('<div class="card"><div class="section-title">📋 Full Report</div>', unsafe_allow_html=True)
            st.markdown("Generate all sections and compile into a downloadable report.")
            full_report_btn = st.button("Generate Full Report", use_container_width=True)
            if full_report_btn:
                with st.spinner("Generating complete analysis report..."):
                    summary_r = generate_summary(paper_text, "General Overview", "Detailed")
                    analysis_r = critical_analysis(paper_text)
                    plain_r = explain_simply(paper_text)

                report = f"""PAPERLENS PRO — FULL ANALYSIS REPORT
{'='*60}
Paper: {uploaded_file.name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*60}

SECTION 1: DETAILED SUMMARY
{'-'*40}
{summary_r}

SECTION 2: CRITICAL ANALYSIS
{'-'*40}
{analysis_r}

SECTION 3: PLAIN LANGUAGE EXPLAINER
{'-'*40}
{plain_r}

{'='*60}
Generated by PaperLens Pro
"""
                st.download_button(
                    "⬇️ Download Full Report (.txt)",
                    data=report,
                    file_name=f"report_{uploaded_file.name.replace('.pdf','')}.txt",
                    mime="text/plain"
                )
                st.success("Report ready — click to download!")
            st.markdown('</div>', unsafe_allow_html=True)

    # ==========
    # TAB 6: COMPARE
    # ==========
    with tab6:
        st.markdown('<p class="section-title">Compare Two Papers Side-by-Side</p>', unsafe_allow_html=True)

        if uploaded_file_2:
            with st.spinner("Loading second paper..."):
                text2, _ = extract_text_from_pdf(uploaded_file_2)

            st.markdown(f"""
            <div style="display:flex; gap:12px; margin-bottom:16px;">
                <div class="card" style="flex:1; margin:0;"><b>Paper A:</b> {uploaded_file.name}</div>
                <div class="card" style="flex:1; margin:0;"><b>Paper B:</b> {uploaded_file_2.name}</div>
            </div>
            """, unsafe_allow_html=True)

            compare_btn = st.button("🆚 Run Comparison Analysis", use_container_width=False)
            if compare_btn:
                with st.spinner("Comparing papers across all dimensions..."):
                    comparison = compare_papers(
                        paper_text, text2,
                        uploaded_file.name, uploaded_file_2.name
                    )
                st.markdown(f"""
                <div class="card card-accent">
                    <div class="section-title">Comparative Analysis</div>
                    <div class="result-box">{comparison}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="text-align:center; padding:48px; color:#484F58;">
                <div style="font-size:2.5rem;">🆚</div>
                <div style="margin-top:12px; font-size:0.9rem;">Upload a second PDF in the sidebar to compare two papers</div>
            </div>
            """, unsafe_allow_html=True)

# =====================
# EMPTY STATE
# =====================
else:
    st.markdown("""
    <div style="text-align:center; padding:80px 20px;">
        <div style="font-size:4rem; margin-bottom:16px;">🔬</div>
        <div style="font-family: 'Playfair Display', serif; font-size:1.6rem; font-weight:700; margin-bottom:8px; color:#E6EDF3;">
            Drop a paper. Understand it instantly.
        </div>
        <div style="color:#8B949E; font-size:0.92rem; max-width:480px; margin:0 auto; line-height:1.65;">
            Upload any academic PDF to get AI-powered summaries, critical analysis, 
            entity extraction, chat Q&A, and export tools — all in one place.
        </div>
        <div style="margin-top:32px; display:flex; gap:12px; justify-content:center; flex-wrap:wrap;">
            <span class="tag-pill">📝 Smart Summaries</span>
            <span class="tag-pill">🔍 Critical Analysis</span>
            <span class="tag-pill">💬 Research Chat</span>
            <span class="tag-pill">📊 Metadata Extraction</span>
            <span class="tag-pill">🆚 Paper Comparison</span>
            <span class="tag-pill">🐦 Tweet Threads</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
  
