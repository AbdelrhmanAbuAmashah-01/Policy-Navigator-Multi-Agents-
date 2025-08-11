import os
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st

# Optional FAISS (fallback to cosine similarity if unavailable)
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# Audio recording/transcription
import sounddevice as sd
import wave
import speech_recognition as sr

# ----------------------------
# Configuration (IDs from your notebook)
# ----------------------------
EMBEDDING_MODEL_ID = "673248d66eb563b2b00f75d1"
LLM_TOOL_ID = "669a63646eb56306647e1091"        # GPT-4 Mini (AiXplain)
SCRAPER_TOOL_ID = "66f423426eb563fa213a3531"     # Scraper tool

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NESTED_PROJECT_DIR = os.path.join(THIS_DIR, "aiXplain_Project")
DATA_DIR = os.path.join(NESTED_PROJECT_DIR, "Data")

# ----------------------------
# API key (kept in code; not shown to users)
# !!! SECURITY NOTE: Do NOT commit real keys to version control.
# Replace the placeholder below with your real key for local runs/deployment.
# ----------------------------
AIXPLAIN_API_KEY = "65fa386c76de9e3a427be836e6d47064966a11b516730ca3ec63c45938879550"  # <-- replace with your real key
os.environ["AIXPLAIN_API_KEY"] = AIXPLAIN_API_KEY

WARNING_PHRASE = (
    "I don't have any idea on this, ask questions that related to the Privacy area of expertise."
)

# ----------------------------
# Presentation helpers
# ----------------------------
import re

def format_agent_output(text: str) -> str:
    """Normalize plain text into Markdown-friendly formatting."""
    if not text:
        return ""
    # Ensure numbered lists start on their own line
    text = re.sub(r"\s*(\d+\.)\s+", r"\n\1 ", text)
    # Add breaks for typical inline numbered items (2..20)
    for i in range(2, 21):
        text = text.replace(f" {i}.", f"\n{i}.")
    # Normalize dashes as bullets
    text = re.sub(r"\s*-\s+", r"\n- ", text)
    # Collapse extra blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def render_markdown_answer(title: str, raw_text: str) -> None:
    """Render a titled markdown answer without private Streamlit APIs."""
    formatted = format_agent_output(raw_text or "")
    try:
        with st.container(border=True):  # Streamlit >= 1.32
            st.markdown(f"**{title}**")
            st.markdown(formatted)
    except TypeError:
        st.markdown(f"#### {title}")
        st.markdown(formatted)

def render_sidebar_help() -> None:
    """Styled sidebar panel explaining what to ask + 5 starter questions."""
    st.sidebar.markdown("""
<div style="
    padding: 1rem;
    border-radius: 12px;
    background: linear-gradient(135deg, #fffde7, #e8f5e9);
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
">
  <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.25rem;">
    <div style="font-size:1.2rem;">🧭</div>
    <h3 style="margin:0;color:#1b5e20;font-weight:700;">What to ask</h3>
  </div>
  <div style="color:#2e7d32;font-size:.95rem;line-height:1.5;">
    <p style="margin:.2rem 0 .6rem 0;">
      Ask about <strong>breach patterns</strong>, <strong>attack types</strong>, <strong>industry trends</strong>, 
      <strong>exposed data</strong>, or <strong>NYSED policy steps</strong>. Keep questions concise.
    </p>
    <p style="margin:.4rem 0 .2rem 0;"><strong>Examples:</strong></p>
    <ol style="margin:.2rem 0 0 1.2rem;">
      <li>Most common <em>breach causes</em> and their proportions?</li>
      <li>Rank <em>cyberattack types</em> by frequency.</li>
      <li>Which <em>industries</em> report the most breaches and why?</li>
      <li>Which <em>personal info types</em> are most often exposed together?</li>
      <li>Outline <em>NYSED incident response</em> for ransomware on student/employee PII.</li>
    </ol>
  </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# API key validation (no UI; silent)
# ----------------------------
def validate_aixplain_key_silently(api_key: str) -> bool:
    if not api_key or api_key.strip() == "" or "YOUR_AIXPLAIN_API_KEY" in api_key:
        return False
    os.environ["AIXPLAIN_API_KEY"] = api_key
    try:
        # This import triggers supplier load; will raise if key invalid.
        from aixplain.factories import ModelFactory  # noqa: F401
        return True
    except Exception:
        return False

# ----------------------------
# Utilities (IO + RAG)
# ----------------------------
def chunk_text(text: str, max_chars: int = 1000, overlap: int = 100) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def read_csv_texts(max_rows: int = 200) -> List[str]:
    import csv
    if not os.path.isdir(DATA_DIR):
        return []
    csv_texts: List[str] = []
    for name in os.listdir(DATA_DIR):
        if name.lower().endswith(".csv"):
            path = os.path.join(DATA_DIR, name)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
                    reader = csv.reader(f)
                    for i, row in enumerate(reader):
                        if i >= max_rows:
                            break
                        row_text = " ".join(cell.strip() for cell in row if cell is not None)
                        if row_text.strip():
                            csv_texts.extend(chunk_text(row_text, max_chars=800, overlap=80))
            except Exception:
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = [line.strip() for line in f if line.strip()]
                    for ln in lines[:max_rows]:
                        csv_texts.extend(chunk_text(ln, max_chars=800, overlap=80))
                except Exception:
                    continue
    return csv_texts

def read_pdf_texts(max_pages: int = 30) -> List[str]:
    try:
        import PyPDF2  # type: ignore
    except Exception:
        return []
    if not os.path.isdir(DATA_DIR):
        return []
    pdf_texts: List[str] = []
    for name in os.listdir(DATA_DIR):
        if name.lower().endswith(".pdf"):
            path = os.path.join(DATA_DIR, name)
            try:
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    num_pages = min(len(reader.pages), max_pages)
                    for page_idx in range(num_pages):
                        try:
                            page_text = reader.pages[page_idx].extract_text() or ""
                            chunks = chunk_text(page_text, max_chars=1000, overlap=100)
                            pdf_texts.extend(chunks)
                        except Exception:
                            continue
            except Exception:
                continue
    return pdf_texts

def extract_embedding_from_aixplain_response(response_obj) -> Optional[List[float]]:
    try:
        if isinstance(response_obj, dict) and "data" in response_obj:
            data = response_obj["data"]
            if isinstance(data, list) and data and isinstance(data[0], dict) and "embedding" in data[0]:
                return list(map(float, data[0]["embedding"]))
            if isinstance(data, list) and data and isinstance(data[0], (float, int)):
                return list(map(float, data))
    except Exception:
        pass
    return None

def build_embeddings_and_index(chunks: List[str]) -> Tuple[np.ndarray, List[str], Optional[object]]:
    """Embed text chunks using aiXplain and build a FAISS index (if available)."""
    if not chunks:
        return np.zeros((0, 0), dtype="float32"), [], None

    # Import here (runtime), *after* key validation succeeded.
    from aixplain.factories import ModelFactory
    embedder = ModelFactory.get(EMBEDDING_MODEL_ID)
    vectors: List[List[float]] = []
    texts: List[str] = []

    for t in chunks:
        try:
            resp = embedder.run(t)
            vec = extract_embedding_from_aixplain_response(resp)
            if vec is not None:
                vectors.append(vec)
                texts.append(t)
        except Exception:
            continue

    if not vectors:
        return np.zeros((0, 0), dtype="float32"), [], None

    embeddings_matrix = np.array(vectors, dtype="float32")

    if FAISS_AVAILABLE:
        try:
            index = faiss.IndexFlatL2(embeddings_matrix.shape[1])
            index.add(embeddings_matrix)
        except Exception:
            index = None
    else:
        index = None

    return embeddings_matrix, texts, index

def _topk_by_cosine(q: np.ndarray, matrix: np.ndarray, k: int) -> List[int]:
    qn = q / (np.linalg.norm(q) + 1e-8)
    mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    sims = (mn @ qn.reshape(-1, 1)).reshape(-1)
    order = np.argsort(sims)[-k:][::-1]
    return order.tolist()

def retrieve_context(query: str, embeddings_matrix: np.ndarray, texts: List[str],
                     index_obj: Optional[object], top_k: int = 5) -> str:
    if not query.strip() or embeddings_matrix.size == 0 or not texts:
        return ""

    # Import here (runtime), *after* key validation succeeded.
    from aixplain.factories import ModelFactory
    embedder = ModelFactory.get(EMBEDDING_MODEL_ID)
    resp = embedder.run(query)
    q_vec = extract_embedding_from_aixplain_response(resp)
    if q_vec is None:
        return ""
    q = np.array([q_vec], dtype="float32")

    if index_obj is not None and FAISS_AVAILABLE:
        try:
            _, I = index_obj.search(q, top_k)
            idxs = I[0]
        except Exception:
            idxs = _topk_by_cosine(q[0], embeddings_matrix, top_k)
    else:
        idxs = _topk_by_cosine(q[0], embeddings_matrix, top_k)

    matched = [texts[i] for i in idxs if 0 <= i < len(texts)]
    return "\n\n".join(matched)

@st.cache_resource(show_spinner=True)
def bootstrap_corpus() -> Tuple[np.ndarray, List[str], Optional[object]]:
    csv_chunks = read_csv_texts(max_rows=200)
    pdf_chunks = read_pdf_texts(max_pages=30)

    combined_chunks = []
    combined_chunks.extend(csv_chunks[:300])
    combined_chunks.extend(pdf_chunks[:300])

    embeddings_matrix, texts, index_obj = build_embeddings_and_index(combined_chunks)
    return embeddings_matrix, texts, index_obj

@st.cache_resource(show_spinner=True)
def get_policy_navigator_agent():
    # Import here (runtime), *after* key validation succeeded.
    from aixplain.factories import AgentFactory
    from aixplain.modules.agent.tool.model_tool import ModelTool
    return AgentFactory.create(
        name="Policy Navigator Agent",
        description="Agent for privacy policy Q&A on the privacy navigator dataset and domain knowledge.",
        instructions=(
            "Answer questions related to privacy, data protection, and policy using the provided context. "
            "If insufficient context, be concise and request a relevant privacy-related question."
        ),
        tools=[ModelTool(model=LLM_TOOL_ID)],
    )

@st.cache_resource(show_spinner=True)
def get_scraper_agent():
    # Import here (runtime), *after* key validation succeeded.
    from aixplain.factories import AgentFactory
    from aixplain.modules.agent.tool.model_tool import ModelTool
    return AgentFactory.create(
        name="Web Scraper Agent",
        description="Agent for live web content extraction",
        instructions="An agent that scrapes web content from a provided URL and returns useful data.",
        tools=[ModelTool(model=SCRAPER_TOOL_ID)],
    )

def run_policy_query(query: str, embeddings_matrix: np.ndarray, texts: List[str],
                     index_obj: Optional[object]) -> str:
    context = retrieve_context(query, embeddings_matrix, texts, index_obj, top_k=5)
    agent = get_policy_navigator_agent()
    prompt = f"Query: {query}\n\nContext:\n{context}"
    resp = agent.run(prompt)
    return str(resp.get("data", {}).get("output", ""))

def run_scrape(url: str) -> str:
    agent = get_scraper_agent()
    resp = agent.run(url)
    return str(resp.get("data", {}).get("output", ""))

def record_audio_to_file(seconds: int = 10, filename: str = "user_audio.wav") -> str:
    st.info(f"Recording for {seconds} seconds... Speak now.")
    audio = sd.rec(int(seconds * 44100), samplerate=44100, channels=1, dtype="int16")
    sd.wait()
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(44100)
        wf.writeframes(audio.tobytes())
    return filename

def transcribe_with_google(audio_file: str) -> Optional[str]:
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.warning("Google Speech Recognition could not understand the audio.")
        return None
    except sr.RequestError as e:
        st.error(f"Speech Recognition API error: {e}")
        return None
    except Exception:
        return None

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Policy Navigator Multi-Agent", page_icon="🧭", layout="centered")
st.title("Policy Navigator Multi-Agent")
st.caption("Minimal app combining Policy Navigator, Web Scraper, and Audio-to-Text agents.")

# Sidebar: helper only (no key shown)
render_sidebar_help()

# Validate API key silently BEFORE any aiXplain import is attempted
AIX_OK = validate_aixplain_key_silently(AIXPLAIN_API_KEY)

if AIX_OK:
    with st.spinner("Bootstrapping corpus and index (first run may take a while)..."):
        EMBEDDINGS_MATRIX, COMBINED_TEXTS, COMBINED_INDEX = bootstrap_corpus()
else:
    EMBEDDINGS_MATRIX = np.zeros((0, 0), dtype="float32")
    COMBINED_TEXTS = []
    COMBINED_INDEX = None
    st.error(
        "The aiXplain API key is missing or invalid. Update the key in the code and rerun the app."
    )

# 1) Policy Navigator Agent — manual text query
st.subheader("1) Policy Navigator Agent")
policy_query = st.text_input(
    "Enter your policy-related question:",
    value="What regulations apply to third-party data sharing?",
    placeholder="Ask about privacy, data protection, breach response, etc."
)
if st.button("Ask Policy Agent", type="primary", disabled=not AIX_OK):
    if not policy_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            answer = run_policy_query(policy_query, EMBEDDINGS_MATRIX, COMBINED_TEXTS, COMBINED_INDEX)
        if WARNING_PHRASE in answer:
            st.warning(answer)
        else:
            render_markdown_answer("Policy Agent Response", answer)

st.divider()

# 2) Web Scraper Agent — URL input
st.subheader("2) Web Scraper Agent")
url = st.text_input(
    "Enter a URL to scrape:",
    value="https://gdpr-info.eu/art-5-gdpr/",
    placeholder="https://..."
)
if st.button("Scrape URL", disabled=not AIX_OK):
    if not url.strip():
        st.warning("Please enter a URL.")
    else:
        with st.spinner("Scraping website content..."):
            scraped = run_scrape(url)
        if WARNING_PHRASE in scraped:
            st.warning(scraped)
        else:
            render_markdown_answer("Scraper Output", scraped)

st.divider()

# 3) Audio-to-Text Policy Agent — record, transcribe, retrieve, respond
st.subheader("3) Audio-to-Text Policy Agent")
col1, col2 = st.columns([1, 1])
with col1:
    duration = st.slider("Record duration (seconds)", min_value=3, max_value=20, value=10, step=1)
with col2:
    st.write("")

if st.button("Record and Ask", disabled=not AIX_OK):
    with st.spinner("Recording..."):
        wav_path = record_audio_to_file(seconds=duration, filename=os.path.join(NESTED_PROJECT_DIR, "user_audio.wav"))
    st.success(f"Audio saved: {wav_path}")

    with st.spinner("Transcribing audio..."):
        transcript = transcribe_with_google(wav_path)

    if not transcript:
        st.warning("No transcription available.")
    else:
        render_markdown_answer("Transcribed Text", transcript)
        with st.spinner("Retrieving context and generating answer..."):
            answer = run_policy_query(transcript, EMBEDDINGS_MATRIX, COMBINED_TEXTS, COMBINED_INDEX)
        if WARNING_PHRASE in answer:
            st.warning(answer)
        else:
            render_markdown_answer("Policy Agent Response", answer)

st.caption("If answers seem off-topic, try a more specific privacy or data protection question.")
