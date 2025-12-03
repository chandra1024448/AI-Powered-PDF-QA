# app.py (FINAL — Offline RAG + Clean Answers + Anti-Repetition + New Title)

import streamlit as st
import pdfplumber
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------------------------------
# UI CONFIG — New Recruiter-Friendly Title
# -----------------------------------------------------
st.set_page_config(page_title="🤖 AI-Powered PDF Question-Answering System (RAG)",
                   page_icon="📘", layout="wide")

st.markdown("""
<h1 style="text-align:center; color:#4F46E5; font-size:40px;">
🤖 AI-Powered PDF Question-Answering System (RAG)
</h1>
<p style="text-align:center; color:#6B7280; font-size:17px;">
Intelligent, offline, context-aware Q&A from any PDF — No API, No Tokens Needed.
</p>
""", unsafe_allow_html=True)



# -----------------------------------------------------
# Load Embedder
# -----------------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()



# -----------------------------------------------------
# Load Offline LLM (Qwen 2.5 - 0.5B Instruct)
# -----------------------------------------------------
@st.cache_resource
def load_llm():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return tok, model

tokenizer, model = load_llm()



# -----------------------------------------------------
# Helpers
# -----------------------------------------------------
def ensure_period(text):
    text = text.strip()
    if not text:
        return text
    if text[-1] not in ".!?":
        return text + "."
    return text


def clean_answer(text, prompt):
    """Remove prompt echo, remove repetition, shorten answer, make clean summary."""
    
    # 1. Remove echoed prompt
    text = text.replace(prompt, "").strip()

    # 2. Remove back-to-back repeated words
    words = text.split()
    cleaned_words = []
    for w in words:
        if cleaned_words and w.lower() == cleaned_words[-1].lower():
            continue
        cleaned_words.append(w)
    text = " ".join(cleaned_words)

    # 3. Remove looping patterns ("scope eligibility scope eligibility…")
    text = re.sub(r'(\b\w+\b)( \1\b)+', r'\1', text, flags=re.IGNORECASE)

    # 4. Keep only the first few meaningful sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    text = " ".join(sentences[:5])  # keep first 5 good sentences

    # 5. Cleanup
    text = re.sub(r"\s+", " ", text).strip()

    return ensure_period(text[:600])  # final trimmed answer




# -----------------------------------------------------
# Session State
# -----------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []



# -----------------------------------------------------
# Upload PDF(s)
# -----------------------------------------------------
uploaded_files = st.file_uploader("📁 Upload PDF (single or multiple)",
                                  type=["pdf"], accept_multiple_files=True)

colA, colB = st.columns([1, 1])
with colA:
    clear_q = st.button("🧹 Clear Question")
with colB:
    clear_all = st.button("🗑️ Clear All History")

if clear_all:
    st.session_state.history = []
    st.experimental_rerun()



if uploaded_files:

    text_pages = []

    # Load multiple PDFs
    for uploaded in uploaded_files:
        st.success(f"📄 Loaded: {uploaded.name}")

        try:
            with pdfplumber.open(uploaded) as pdf:
                for i, p in enumerate(pdf.pages):
                    text_pages.append({"page": f"{uploaded.name} - p{i+1}",
                                       "text": p.extract_text() or ""})
        except:
            reader = PdfReader(uploaded)
            for i, p in enumerate(reader.pages):
                text_pages.append({"page": f"{uploaded.name} - p{i+1}",
                                   "text": p.extract_text() or ""})


    full_text = "\n".join([p["text"] for p in text_pages])
    st.info(f"Total Pages: {len(text_pages)} | Words: {len(full_text.split())}")
    st.success("✅ PDF processed successfully!")



    # -----------------------------------------------------
    # Chunking
    # -----------------------------------------------------
    chunk_size_words = 500
    chunks, pages_of_chunk = [], []
    cur, cur_pages = "", []

    for pg in text_pages:
        txt = pg["text"].strip()
        if not txt:
            continue

        if len((cur + " " + txt).split()) > chunk_size_words and cur:
            chunks.append(cur.strip())
            pages_of_chunk.append(cur_pages[:])
            cur = txt
            cur_pages = [pg["page"]]
        else:
            cur = (cur + "\n" + txt).strip()
            cur_pages.append(pg["page"])

    if cur.strip():
        chunks.append(cur.strip())
        pages_of_chunk.append(cur_pages[:])



    # -----------------------------------------------------
    # Embeddings + FAISS
    # -----------------------------------------------------
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))



    # -----------------------------------------------------
    # Ask Question
    # -----------------------------------------------------
    st.markdown("### 💬 Ask your question:")
    question = st.text_input("Type here...")

    if clear_q:
        st.experimental_rerun()

    ask = st.button("🔍 Get Answer")

    if ask and question.strip():
        with st.spinner("Thinking…"):

            # Retrieve top chunks
            q_emb = embedder.encode([question], convert_to_numpy=True).astype("float32")
            k = min(4, len(chunks))
            D, I = index.search(q_emb, k)

            ctx = ""
            used_pages = []

            for idx in I[0]:
                ctx += chunks[idx] + "\n"
                used_pages.extend(pages_of_chunk[idx])

            # Build the prompt
            prompt = (
                f"Use the following context to answer the question clearly and simply.\n"
                f"If the answer is not found, say: 'No direct answer found in document.'\n\n"
                f"Context:\n{ctx}\n\n"
                f"Question: {question}\nAnswer:"
            )

            # Generate answer
            inputs = tokenizer(prompt, return_tensors="pt")
            output = model.generate(
                **inputs,
                max_new_tokens=400,
                repetition_penalty=1.1,   # avoids loops
                temperature=0.2,          # keeps answers accurate
            )
            raw = tokenizer.decode(output[0], skip_special_tokens=True)

            # Clean answer
            answer = clean_answer(raw, prompt)

            if used_pages:
                answer += f" (pages: {', '.join(used_pages[:5])})."

            st.session_state.history.append({"q": question, "a": answer})



    # -----------------------------------------------------
    # Display History
    # -----------------------------------------------------
    if st.session_state.history:
        st.markdown("### 🔎 Chat History")
        for item in reversed(st.session_state.history):
            st.markdown(f"**Q:** {item['q']}")
            st.markdown(f"**A:** {item['a']}")
            st.write("---")


else:
    st.info("Upload a PDF to begin.")



# -----------------------------------------------------
# Footer
# -----------------------------------------------------
st.markdown(
    "<p style='text-align:center; color:#888888;'>Built with Streamlit + FAISS + Qwen2.5-0.5B-Instruct (Offline RAG).</p>",
    unsafe_allow_html=True
)
