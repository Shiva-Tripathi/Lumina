import streamlit as st
import os
import tempfile
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Lumina — AI Document Assistant",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

:root {
  --bg:       #0d0d0f;
  --surface:  #141417;
  --card:     #1a1a1f;
  --border:   rgba(255,255,255,0.07);
  --border2:  rgba(255,255,255,0.12);
  --accent:   #7c6aff;
  --accent2:  #a78bfa;
  --text:     #f0eff4;
  --muted:    #7f7e8c;
  --muted2:   #a3a2b0;
  --success:  #34d399;
  --user-bg:  #7c6aff;
  --ai-bg:    #1e1e26;
}

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 2rem !important; max-width: 100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: rgba(124,106,255,0.04) !important;
  border: 1.5px dashed rgba(124,106,255,0.35) !important;
  border-radius: 14px !important;
  padding: 0.5rem !important;
  transition: all 0.25s ease !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--accent) !important;
  background: rgba(124,106,255,0.08) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] p { color: var(--muted) !important; }

/* ── Primary button ── */
.stButton > button {
  background: linear-gradient(135deg, #7c6aff, #6d5ce7) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.88rem !important;
  padding: 0.6rem 1.4rem !important;
  box-shadow: 0 0 20px rgba(124,106,255,0.3) !important;
  transition: all 0.2s ease !important;
}
.stButton > button:hover {
  box-shadow: 0 0 30px rgba(124,106,255,0.5) !important;
  transform: translateY(-1px) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
  background: var(--card) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 16px !important;
  transition: all 0.2s ease !important;
}
[data-testid="stChatInput"]:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(124,106,255,0.15) !important;
}
[data-testid="stChatInput"] textarea {
  background: transparent !important;
  color: var(--text) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.95rem !important;
}
[data-testid="stChatInput"] button { background: var(--accent) !important; border-radius: 10px !important; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted) !important; }

/* ── Progress ── */
.stProgress > div > div { background: linear-gradient(90deg,#7c6aff,#a78bfa) !important; border-radius:99px !important; }
.stProgress > div { background: rgba(255,255,255,0.06) !important; border-radius:99px !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 0.8rem 1rem !important;
}
[data-testid="stMetricLabel"] { font-size:0.75rem !important; color:var(--muted) !important; letter-spacing:0.06em !important; text-transform:uppercase !important; }
[data-testid="stMetricValue"] { font-family:'Syne',sans-serif !important; font-size:1.6rem !important; color:var(--text) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
  background: rgba(255,255,255,0.025) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}
[data-testid="stExpander"] summary { color:var(--muted2) !important; font-size:0.82rem !important; }

/* ── Toggle ── */
[data-testid="stToggle"] span { background:var(--border2) !important; }

/* ── Scrollbars ── */
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.1); border-radius:99px; }

/* ═══ CUSTOM HTML COMPONENTS ═══ */

.sidebar-logo { padding:2rem 1.5rem 1.2rem; border-bottom:1px solid var(--border); margin-bottom:1.2rem; }
.logo-mark { font-family:'Syne',sans-serif; font-size:1.65rem; font-weight:800; letter-spacing:-0.02em;
  background:linear-gradient(135deg,#a78bfa,#7c6aff,#818cf8);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.logo-tag { font-size:0.7rem; color:var(--muted) !important; letter-spacing:0.09em; text-transform:uppercase; margin-top:0.2rem; }

.model-badge { display:flex; align-items:center; gap:8px; background:rgba(255,255,255,0.04);
  border:1px solid var(--border); border-radius:8px; padding:0.55rem 0.8rem;
  font-size:0.78rem; color:var(--muted2); margin-bottom:0.8rem; }
.model-dot { width:7px; height:7px; border-radius:50%; background:var(--success);
  box-shadow:0 0 8px rgba(52,211,153,0.6); flex-shrink:0; }

.section-label { font-size:0.68rem; letter-spacing:0.1em; text-transform:uppercase;
  color:var(--muted) !important; margin-bottom:0.5rem; display:block; }

.doc-card { background:linear-gradient(135deg,rgba(124,106,255,0.08),rgba(129,140,248,0.03));
  border:1px solid rgba(124,106,255,0.2); border-radius:12px; padding:0.9rem 1rem; margin-bottom:1rem; }
.doc-name { font-size:0.85rem; font-weight:500; color:var(--text); margin-bottom:0.5rem;
  display:flex; align-items:center; gap:6px; }
.doc-stats { display:flex; gap:1rem; }
.doc-stat { font-size:0.75rem; color:var(--muted2); }
.doc-stat span { color:var(--accent2); font-weight:600; }

.lumina-hr { border:none; border-top:1px solid var(--border); margin:1rem 0; }

.lumina-hero { padding:2.2rem 0 1.2rem; position:relative; }
.lumina-hero::before { content:''; position:absolute; top:-60px; left:-100px; width:500px; height:320px;
  background:radial-gradient(ellipse,rgba(124,106,255,0.09) 0%,transparent 70%); pointer-events:none; z-index:0; }
.eyebrow { font-size:0.7rem; letter-spacing:0.14em; text-transform:uppercase; color:var(--accent2);
  margin-bottom:0.55rem; display:flex; align-items:center; gap:8px; }
.eyebrow::before { content:''; display:inline-block; width:18px; height:1.5px; background:var(--accent); }
.hero-title { font-family:'Syne',sans-serif; font-size:2.9rem; font-weight:800; line-height:1.08;
  letter-spacing:-0.04em; color:var(--text); margin:0 0 0.5rem; }
.accent-word { background:linear-gradient(135deg,#a78bfa,#818cf8);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.hero-sub { color:var(--muted2); font-size:0.95rem; max-width:460px; line-height:1.65; margin:0; }

.status-pill { display:inline-flex; align-items:center; gap:7px; padding:0.35rem 0.9rem;
  border-radius:99px; font-size:0.78rem; font-weight:500; border:1px solid transparent; }
.status-pill.idle { background:rgba(255,255,255,0.04); border-color:var(--border); color:var(--muted); }
.status-pill.idle .dot { background:var(--muted); }
.status-pill.ready { background:rgba(52,211,153,0.08); border-color:rgba(52,211,153,0.22); color:var(--success); }
.status-pill.ready .dot { background:var(--success); animation:blink 2s ease-in-out infinite; }
.status-pill .dot { width:6px; height:6px; border-radius:50%; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

.step-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:7px; margin:0.8rem 0; }
.step-card { background:var(--card); border:1px solid var(--border); border-radius:9px;
  padding:0.65rem 0.6rem; text-align:center; font-size:0.73rem; color:var(--muted); transition:all 0.3s ease; }
.step-card.active { border-color:var(--accent); background:rgba(124,106,255,0.1);
  color:var(--accent2); box-shadow:0 0 14px rgba(124,106,255,0.2); }
.step-card.done { border-color:rgba(52,211,153,0.3); background:rgba(52,211,153,0.06); color:var(--success); }
.step-icon { font-size:1rem; margin-bottom:0.25rem; display:block; }

.chat-area { display:flex; flex-direction:column; gap:1rem; min-height:52vh; max-height:52vh;
  overflow-y:auto; padding:0.25rem 0.1rem; scrollbar-width:thin; }

.bubble-row { display:flex; gap:10px; animation:fadeSlide 0.26s ease; }
.bubble-row.user { flex-direction:row-reverse; }
@keyframes fadeSlide { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }

.avatar { width:30px; height:30px; border-radius:50%; display:flex; align-items:center;
  justify-content:center; font-size:0.7rem; font-weight:600; flex-shrink:0; margin-top:2px; }
.avatar.ai { background:linear-gradient(135deg,#7c6aff,#a78bfa); color:#fff;
  box-shadow:0 0 12px rgba(124,106,255,0.4); }
.avatar.user { background:rgba(255,255,255,0.08); color:var(--muted2); border:1px solid var(--border2); }

.bubble { max-width:72%; padding:0.85rem 1.1rem; border-radius:16px;
  font-size:0.93rem; line-height:1.65; }
.bubble.user { background:var(--user-bg); color:#fff; border-bottom-right-radius:4px; }
.bubble.ai { background:var(--ai-bg); color:var(--text); border:1px solid var(--border);
  border-bottom-left-radius:4px; }

.empty-state { display:flex; flex-direction:column; align-items:center; justify-content:center;
  min-height:50vh; gap:1rem; text-align:center; }
.empty-icon { width:62px; height:62px; background:rgba(124,106,255,0.1);
  border:1px solid rgba(124,106,255,0.22); border-radius:18px;
  display:flex; align-items:center; justify-content:center; font-size:1.6rem; }
.empty-h { font-family:'Syne',sans-serif; font-size:1.25rem; font-weight:700; color:var(--text); margin:0; }
.empty-p { color:var(--muted) !important; font-size:0.88rem; max-width:280px; margin:0; line-height:1.6; }

.chips { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin-top:0.3rem; }
.chip { background:rgba(255,255,255,0.04); border:1px solid var(--border2); border-radius:99px;
  padding:0.35rem 0.85rem; font-size:0.79rem; color:var(--muted2); }

.source-card { background:rgba(255,255,255,0.025); border:1px solid var(--border);
  border-left:2px solid var(--accent); border-radius:8px; padding:0.65rem 0.85rem;
  margin-bottom:0.5rem; font-size:0.8rem; color:var(--muted2); line-height:1.55; }
.src-label { font-size:0.68rem; color:var(--accent2); font-weight:500;
  letter-spacing:0.05em; text-transform:uppercase; margin-bottom:0.3rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────
for k, v in {"vectorstore": None, "retriever": None, "chat_history": [], "doc_meta": {}}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embed_model():
    return MistralAIEmbeddings()

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(model="mistral-small-2506")

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are Lumina. Answer the question using ONLY the provided context.
Be concise and direct. Do not add information not present in the context.
Start your answer immediately without preamble."""),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
])

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    docs = PyPDFLoader(tmp_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200).split_documents(docs)
    persist_dir = os.path.join(tempfile.gettempdir(), "lumina_chroma")
    vs = Chroma.from_documents(documents=chunks, embedding=get_embed_model(), persist_directory=persist_dir)
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7})
    os.unlink(tmp_path)
    return {"vectorstore": vs, "retriever": retriever,
            "pages": len(docs), "chunks": len(chunks), "filename": uploaded_file.name}

def ask(query):
    docs = st.session_state.retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])
    response = get_llm().invoke(PROMPT.invoke({"context": context, "question": query}))
    return {"answer": response.content, "sources": [d.page_content[:265] + "…" for d in docs]}

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div class="logo-mark">✦ Lumina</div>
      <div class="logo-tag">AI Document Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="model-badge">
      <div class="model-dot"></div>
      mistral-small-2506 · RAG pipeline
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="section-label">📄 Document</span>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Drop PDF here", type=["pdf"], label_visibility="collapsed")

    if uploaded_file:
        if st.button("⚡  Process Document", use_container_width=True):
            steps_ph = st.empty()
            prog_ph   = st.empty()

            STEPS = [("📖","Reading"), ("✂️","Chunking"), ("🔮","Embedding"), ("💾","Indexing")]

            def draw_steps(active, done):
                h = '<div class="step-grid">'
                for i, (ic, lb) in enumerate(STEPS):
                    cls = "done" if i < done else ("active" if i == active else "step-card")
                    h += f'<div class="step-card {cls}"><span class="step-icon">{"✓" if i < done else ic}</span>{lb}</div>'
                return h + "</div>"

            prog = prog_ph.progress(0, text="Initialising…")
            steps_ph.markdown(draw_steps(0, 0), unsafe_allow_html=True)
            time.sleep(0.3); prog.progress(15, text="Reading PDF…")
            time.sleep(0.4); prog.progress(35, text="Chunking text…")
            steps_ph.markdown(draw_steps(1, 1), unsafe_allow_html=True)
            time.sleep(0.3); prog.progress(55, text="Generating embeddings…")
            steps_ph.markdown(draw_steps(2, 2), unsafe_allow_html=True)
            result = process_pdf(uploaded_file)
            prog.progress(85, text="Building index…")
            steps_ph.markdown(draw_steps(3, 3), unsafe_allow_html=True)
            time.sleep(0.5); prog.progress(100, text="Complete!")
            steps_ph.markdown(draw_steps(4, 4), unsafe_allow_html=True)
            time.sleep(0.6)
            prog_ph.empty(); steps_ph.empty()

            st.session_state.update({
                "vectorstore": result["vectorstore"],
                "retriever": result["retriever"],
                "doc_meta": result,
                "chat_history": [],
            })
            st.success("✓ Document ready!")
            st.rerun()

    if st.session_state.doc_meta:
        m = st.session_state.doc_meta
        n = m["filename"]
        short = (n[:26] + "…") if len(n) > 28 else n
        st.markdown(f"""
        <div class="doc-card">
          <div class="doc-name">📄 {short}</div>
          <div class="doc-stats">
            <div class="doc-stat"><span>{m['pages']}</span> pages</div>
            <div class="doc-stat"><span>{m['chunks']}</span> chunks</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="lumina-hr">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">⚙️ Options</span>', unsafe_allow_html=True)
    show_sources = st.toggle("Show source excerpts", value=True)
    st.markdown('<hr class="lumina-hr">', unsafe_allow_html=True)

    if st.button("🗑  Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("""
    <div style="padding:1.2rem 0 0.3rem;font-size:0.7rem;color:#3d3c4a;text-align:center;line-height:1.9;">
      LangChain · ChromaDB · MistralAI · Streamlit
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
is_ready = st.session_state.retriever is not None

# Hero
st.markdown("""
<div class="lumina-hero">
  <div class="eyebrow">AI Document Intelligence</div>
  <div class="hero-title">Ask your <span class="accent-word">documents</span><br>anything.</div>
  <p class="hero-sub">Upload a PDF and have a natural conversation with its contents — powered by retrieval-augmented generation.</p>
</div>
""", unsafe_allow_html=True)

# Status + metrics
col_s, col_p, col_c, col_gap = st.columns([2.2, 1, 1, 3.5])
with col_s:
    if is_ready:
        st.markdown('<div class="status-pill ready"><span class="dot"></span>Document loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill idle"><span class="dot"></span>No document</div>', unsafe_allow_html=True)
if is_ready:
    m = st.session_state.doc_meta
    with col_p: st.metric("Pages", m["pages"])
    with col_c: st.metric("Chunks", m["chunks"])

st.markdown('<hr class="lumina-hr" style="margin:1.2rem 0;">', unsafe_allow_html=True)

# ── Chat area ──
if not is_ready:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">✦</div>
      <div class="empty-h">Ready when you are</div>
      <p class="empty-p">Upload a PDF from the sidebar to start your AI-powered document conversation.</p>
      <div class="chips">
        <span class="chip">📘 Research papers</span>
        <span class="chip">📊 Business reports</span>
        <span class="chip">📖 Textbooks</span>
        <span class="chip">📋 Legal docs</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

elif not st.session_state.chat_history:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">💬</div>
      <div class="empty-h">Start the conversation</div>
      <p class="empty-p">Your document is indexed and ready. Ask anything below.</p>
      <div class="chips">
        <span class="chip">Summarize this document</span>
        <span class="chip">What are the key findings?</span>
        <span class="chip">What is the main topic?</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    html = '<div class="chat-area">'
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            html += f"""
            <div class="bubble-row user">
              <div class="avatar user">You</div>
              <div class="bubble user">{msg['content']}</div>
            </div>"""
        else:
            html += f"""
            <div class="bubble-row ai">
              <div class="avatar ai">✦</div>
              <div class="bubble ai">{msg['content']}</div>
            </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    # Source excerpts
    if show_sources:
        last_src = next(
            (m.get("sources") for m in reversed(st.session_state.chat_history) if m.get("role") == "assistant"),
            None,
        )
        if last_src:
            with st.expander(f"📚 {len(last_src)} source passages retrieved"):
                for i, s in enumerate(last_src, 1):
                    st.markdown(f'<div class="source-card"><div class="src-label">Excerpt {i}</div>{s}</div>',
                                unsafe_allow_html=True)

# ── Input ──
st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

if is_ready:
    query = st.chat_input("Ask anything about your document…")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.spinner("Thinking…"):
            result = ask(query)
        st.session_state.chat_history.append({
            "role": "assistant", "content": result["answer"], "sources": result["sources"],
        })
        st.rerun()
else:
    st.chat_input("Upload a PDF first to start chatting…", disabled=True)