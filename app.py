"""
app.py

Daraz Customer Support Operations Assistant
--------------------------------------------
A Streamlit chat app that:
  - Loads a PRE-BUILT FAISS index + metadata.json from ./faiss_index
    (built earlier by ingest.py). It NEVER re-reads or re-embeds the
    source PDFs — only the user's live query is embedded at runtime.
  - Lets the user restrict retrieval to one or more knowledge-base
    sections (returns, delivery, refunds, sellers, payments,
    customer_support) via the sidebar.
  - Sends retrieved chunks + the user's question to a Groq-hosted LLM
    (model: openai/gpt-oss-120b) to generate a grounded answer.
  - Reads the Groq API key from Streamlit secrets (st.secrets), never
    from a visible text input.
  - Voice input: user can record a question with the mic; the audio is
    transcribed to text using Groq's Whisper model (whisper-large-v3),
    then goes through the exact same retrieval + LLM pipeline as typed text.
  - Voice output (optional, toggle in sidebar): the assistant's answer is
    converted to speech (gTTS) and played back automatically.

Run:
    streamlit run app.py

Requires a Streamlit secret:
    .streamlit/secrets.toml
        GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
INDEX_DIR = Path("faiss_index")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # must match the model used in ingest.py
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K_DEFAULT = 5
FETCH_MULTIPLIER = 8   # over-fetch this many x top_k, then filter by department

ALL_SECTIONS = [
    "returns",
    "delivery",
    "refunds",
    "sellers",
    "payments",
    "customer_support",
]

DARAZ_ORANGE = "#F85606"
DARAZ_DARK = "#1A1A1A"

WHISPER_MODEL = "whisper-large-v3"   # Groq speech-to-text model
TTS_CHAR_LIMIT = 1500                # keep gTTS requests reasonable in length


# --------------------------------------------------------------------------
# Page config + branding
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Daraz Customer Support Assistant",
    page_icon="🛍️",
    layout="wide",
)

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: #FAFAFA;
        }}
        [data-testid="stSidebar"] {{
            background-color: {DARAZ_DARK};
        }}
        [data-testid="stSidebar"] * {{
            color: #FFFFFF !important;
        }}
        .daraz-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 14px 20px;
            background: linear-gradient(90deg, {DARAZ_ORANGE} 0%, #FF8A3D 100%);
            border-radius: 10px;
            margin-bottom: 18px;
        }}
        .daraz-header h1 {{
            color: white;
            font-size: 22px;
            margin: 0;
        }}
        .daraz-header p {{
            color: #FFEDE0;
            margin: 0;
            font-size: 13px;
        }}
        .source-tag {{
            display: inline-block;
            background: #FFF1E8;
            color: {DARAZ_ORANGE};
            border: 1px solid {DARAZ_ORANGE};
            border-radius: 6px;
            padding: 2px 8px;
            font-size: 12px;
            margin: 2px 4px 2px 0;
        }}
        .stChatMessage {{
            border-radius: 10px;
        }}
        div[data-testid="stChatInput"] textarea {{
            border: 1.5px solid {DARAZ_ORANGE} !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="daraz-header">
        <div style="font-size:30px;">🛍️</div>
        <div>
            <h1>Daraz Customer Support Operations Assistant</h1>
            <p>Ask about returns, delivery, refunds, sellers, payments & customer support policies</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached resource loaders — run once per session
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="Loading knowledge base index...")
def load_index_and_metadata():
    index_path = INDEX_DIR / "index.faiss"
    metadata_path = INDEX_DIR / "metadata.json"

    if not index_path.exists() or not metadata_path.exists():
        return None, None

    index = faiss.read_index(str(index_path))
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


@st.cache_resource(show_spinner=False)
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


embedder = load_embedding_model()
faiss_index, metadata = load_index_and_metadata()
groq_client = load_groq_client()


# --------------------------------------------------------------------------
# Sidebar — knowledge base sections + settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📚 Knowledge Base Sections")
    st.caption("Restrict search to specific sections, or leave all selected to search everything.")

    available_sections = sorted(set(m["department"] for m in metadata)) if metadata else ALL_SECTIONS

    selected_sections = []
    for section in available_sections:
        checked = st.checkbox(section.replace("_", " ").title(), value=True, key=f"sec_{section}")
        if checked:
            selected_sections.append(section)

    st.divider()

    with st.expander("⚙️ Advanced settings"):
        top_k = st.slider("Chunks to retrieve", min_value=2, max_value=10, value=TOP_K_DEFAULT)
        show_sources = st.checkbox("Show retrieved sources", value=True)

    st.divider()
    st.markdown("### 🎙️ Voice")
    voice_answers_enabled = st.toggle("🔊 Speak answers out loud", value=False)

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Daraz Ops Assistant · powered by Groq (openai/gpt-oss-120b)")


# --------------------------------------------------------------------------
# Guard rails — missing index or missing API key
# --------------------------------------------------------------------------
if faiss_index is None or metadata is None:
    st.error(
        "No pre-built FAISS index found in `faiss_index/`. "
        "Run `ingest.py` first to build the index, then restart this app."
    )
    st.stop()

if groq_client is None:
    st.error(
        "No Groq API key found. Add `GROQ_API_KEY` to your Streamlit secrets "
        "(`.streamlit/secrets.toml` or the Secrets panel on Streamlit Cloud)."
    )
    st.stop()

if not selected_sections:
    st.warning("Select at least one knowledge base section from the sidebar to search.")
    st.stop()


# --------------------------------------------------------------------------
# Retrieval — embed query only, search the pre-built index, filter by section
# --------------------------------------------------------------------------
def retrieve_chunks(query: str, sections: list, k: int):
    query_vec = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")

    # Over-fetch so that after filtering by department we still have enough results
    fetch_k = min(k * FETCH_MULTIPLIER, faiss_index.ntotal)
    scores, indices = faiss_index.search(query_vec, fetch_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = metadata[idx]
        if item["department"] in sections:
            results.append({**item, "score": float(score)})
        if len(results) >= k:
            break

    return results


def transcribe_audio(audio_bytes: bytes) -> str | None:
    """
    Send recorded mic audio to Groq's Whisper model and return the
    transcribed text. Returns None (and shows an error) on failure.

    language="en" forces Whisper to write the transcription in Roman/Latin
    script (phonetic English spelling), even if the user speaks Urdu/Hindi
    words — otherwise Whisper auto-detects the language and can switch to
    Devanagari/Urdu script mid-sentence.
    """
    try:
        transcription = groq_client.audio.transcriptions.create(
            file=("voice_question.wav", audio_bytes),
            model=WHISPER_MODEL,
            language="en",
            response_format="text",
        )
        # Some SDK versions return a plain string, others an object with .text
        text = transcription if isinstance(transcription, str) else getattr(transcription, "text", "")
        text = text.strip()
        return text if text else None
    except Exception as e:
        st.error(f"Voice transcription failed: {e}")
        return None


def text_to_speech(answer_text: str) -> bytes | None:
    """
    Convert the assistant's answer text to speech (mp3 bytes) using gTTS.
    Truncates very long answers so the audio stays short and reliable.
    """
    try:
        clipped = answer_text[:TTS_CHAR_LIMIT]
        tts = gTTS(text=clipped, lang="en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        st.warning(f"Could not generate voice answer: {e}")
        return None


def build_context(chunks: list) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(
            f"[Source {i} | Section: {c['department']} | File: {c['source_file']}]\n{c['chunk_text']}"
        )
    return "\n\n---\n\n".join(blocks)


SYSTEM_PROMPT = """You are the Daraz Customer Support Operations Assistant, an internal tool used by \
Daraz support agents and operations staff to quickly answer questions about company policy on \
returns, delivery, refunds, sellers, payments, and customer support.

Rules:
- Answer ONLY using the information in the provided context chunks below.
- If the context does not contain enough information to answer confidently, say so clearly and \
suggest which section or team to check with instead — do not invent policy details.
- Be concise, operational, and practical — the reader is an agent who needs to act on this, not a \
customer reading marketing copy.
- When relevant, mention which section(s) (e.g. Returns, Refunds) the answer is based on.
- Do not mention "chunks", "embeddings", "FAISS", or any internal system machinery in your answer.
"""


def generate_answer(question: str, context: str, history: list) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # include a little recent chat history for continuity
    for m in history[-4:]:
        messages.append({"role": m["role"], "content": m["content"]})

    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        }
    )

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    return completion.choices[0].message.content


# --------------------------------------------------------------------------
# Chat state + history rendering
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            tags = "".join(
                f'<span class="source-tag">{s["department"]} · {s["source_file"]}</span>'
                for s in msg["sources"]
            )
            st.markdown(tags, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Chat input — typed text OR voice (mic → Groq Whisper → text)
# --------------------------------------------------------------------------
mic_col, hint_col = st.columns([1, 9])
with mic_col:
    audio = mic_recorder(
        start_prompt="🎙️ Speak",
        stop_prompt="⏹️ Stop",
        just_once=True,
        use_container_width=True,
        key="voice_recorder",
    )
with hint_col:
    st.caption("Tap the mic to ask by voice, or type below.")

typed_question = st.chat_input("Ask about returns, delivery, refunds, sellers, payments...")

# Resolve the actual question from whichever input source fired this run.
# Typed input takes priority if somehow both fire in the same run.
user_question = None
if typed_question:
    user_question = typed_question
elif audio and audio.get("bytes"):
    with st.spinner("Transcribing your voice..."):
        user_question = transcribe_audio(audio["bytes"])
    if user_question:
        st.info(f"🎙️ Heard: \u201c{user_question}\u201d")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            chunks = retrieve_chunks(user_question, selected_sections, top_k)

        if not chunks:
            answer = (
                "I couldn't find anything relevant in the selected section(s). "
                "Try selecting more sections in the sidebar, or rephrase your question."
            )
            st.markdown(answer)
            if voice_answers_enabled:
                audio_bytes = text_to_speech(answer)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": []})
        else:
            context = build_context(chunks)
            with st.spinner("Generating answer..."):
                answer = generate_answer(user_question, context, st.session_state.messages)

            st.markdown(answer)

            if voice_answers_enabled:
                audio_bytes = text_to_speech(answer)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)

            if show_sources:
                tags = "".join(
                    f'<span class="source-tag">{c["department"]} · {c["source_file"]}</span>'
                    for c in chunks
                )
                st.markdown(tags, unsafe_allow_html=True)
                with st.expander("View retrieved excerpts"):
                    for i, c in enumerate(chunks, start=1):
                        st.markdown(f"**{i}. {c['source_file']}** ({c['department']}, score: {c['score']:.3f})")
                        st.caption(c["chunk_text"][:500] + ("..." if len(c["chunk_text"]) > 500 else ""))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": [{"department": c["department"], "source_file": c["source_file"]} for c in chunks],
                }
            )
