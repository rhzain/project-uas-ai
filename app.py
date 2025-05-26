import streamlit as st
import json
import faiss
import numpy as np
import os
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# --- 0. Konfigurasi Awal & Muat Model ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY tidak ditemukan. Harap atur di file .env atau environment variable.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)

# Model Embedding (HARUS SAMA dengan yang di Fase 1)
# Jika di Fase 1 pakai 'all-MiniLM-L6-v2', di sini juga pakai 'all-MiniLM-L6-v2'
MODEL_EMBEDDING_NAME = 'all-MiniLM-L6-v2' # Dimensi 384

# Path ke file hasil Fase 1
FAISS_INDEX_PATH = "vector_store.index"
TEXT_CHUNKS_PATH = "processed_chunks_with_metadata.json"

@st.cache_resource # Cache resource agar tidak load ulang terus
def load_all_resources():
    print("Memuat resources RAG...")
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(TEXT_CHUNKS_PATH, "r", encoding="utf-8") as f:
            text_chunks_list = json.load(f)
        
        print(f"Memuat model embedding untuk query: {MODEL_EMBEDDING_NAME}...")
        query_embedding_model = SentenceTransformer(MODEL_EMBEDDING_NAME)
        print("Model embedding query dimuat.")

        print("Memuat model LLM Gemini...")
        llm_gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest") # Ganti dengan model Gemini yang Anda inginkan
        print("Model LLM Gemini dimuat.")
        
        st.success(f"Resources RAG berhasil dimuat: FAISS index ({index.ntotal} vektor), Text chunks ({len(text_chunks_list)}).")
        return index, text_chunks_list, query_embedding_model, llm_gemini_model
    except Exception as e:
        st.error(f"Gagal memuat resources RAG: {e}")
        print(f"Error detail saat load_all_resources: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

# Muat semua resource saat aplikasi dimulai
faiss_index, text_chunks, query_embed_model, llm_model = load_all_resources()

# --- 1. Fungsi untuk Embed Pertanyaan Pengguna ---
def get_query_embedding_vector(query_text):
    if query_embed_model is None:
        st.error("Model embedding query belum siap.")
        return None
    try:
        return query_embed_model.encode([query_text])[0]
    except Exception as e:
        st.error(f"Error saat membuat embedding query: {e}")
        return None

# --- 2. Fungsi untuk Retrieval dari FAISS ---
def retrieve_from_faiss(query_embedding_vector, k=3):
    if faiss_index is None or query_embedding_vector is None or not text_chunks:
        return []
    try:
        query_np = np.array([query_embedding_vector]).astype('float32')
        
        # Cek dimensi (PENTING!)
        if faiss_index.d != query_np.shape[1]:
            st.error(f"Dimensi tidak cocok! Index FAISS: {faiss_index.d}, Query: {query_np.shape[1]}")
            return []
            
        distances, indices = faiss_index.search(query_np, k)
        
        retrieved_chunks = []
        for i in indices[0]:
            if i != -1 and 0 <= i < len(text_chunks): # FAISS bisa mengembalikan -1
                retrieved_chunks.append(text_chunks[i])
        return retrieved_chunks
    except Exception as e:
        st.error(f"Error saat retrieval dari FAISS: {e}")
        print(f"Error detail saat retrieve_from_faiss: {e}")
        import traceback
        traceback.print_exc()
        return []

# --- 3. Fungsi untuk Menghasilkan Jawaban dengan LLM (RAG) ---
def generate_llm_response(user_query, retrieved_context_chunks):
    if llm_model is None:
        st.error("Model LLM Gemini belum siap.")
        return "Maaf, layanan chatbot sedang tidak tersedia."
        
    if not retrieved_context_chunks:
        # Opsi jika tidak ada konteks: jawab langsung atau beri tahu tidak ada info
        # prompt_no_context = f"Jawab pertanyaan berikut berdasarkan pengetahuan umum Anda: \"{user_query}\""
        # response = llm_model.generate_content(prompt_no_context)
        return "Maaf, saya tidak menemukan informasi yang relevan dalam materi yang tersedia untuk menjawab pertanyaan Anda."

    context_string = "\n\n---\n\n".join(retrieved_context_chunks)
    
    prompt_with_context = f"""Anda adalah chatbot edukasi AI yang sangat membantu.
Berdasarkan konteks materi yang diberikan di bawah ini:
---
{context_string}
---
Jawablah pertanyaan mahasiswa berikut dengan jelas, akurat, dan hanya berdasarkan konteks yang diberikan: "{user_query}"
Jika informasi tidak ada dalam konteks, katakan dengan sopan bahwa Anda tidak dapat menemukan jawabannya dalam materi tersebut. Jangan mencoba membuat jawaban sendiri di luar konteks.
"""
    try:
        response = llm_model.generate_content(prompt_with_context)
        if hasattr(response, 'text'):
            return response.text
        elif response.parts:
            return "".join(part.text for part in response.parts)
        else:
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                return f"Tidak bisa menghasilkan jawaban. Feedback dari LLM: {response.prompt_feedback}"
            return "Tidak bisa menghasilkan jawaban (format respons LLM tidak dikenali)."
    except Exception as e:
        st.error(f"Error saat menghasilkan jawaban dengan LLM: {e}")
        print(f"Error detail saat generate_llm_response: {e}")
        import traceback
        traceback.print_exc()
        return "Terjadi kesalahan internal saat mencoba menghasilkan jawaban."

# --- 4. Antarmuka Streamlit ---
st.title("🤖 Chatbot Edukasi AI (RAG + Gemini)")
st.caption("Tanyakan apa saja tentang materi kuliah Anda!")

# Inisialisasi histori chat di session state Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan pesan-pesan dari histori
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Terima input dari pengguna
if user_input_query := st.chat_input("Ketik pertanyaan Anda di sini..."):
    # Tambahkan pesan pengguna ke histori dan tampilkan
    st.session_state.messages.append({"role": "user", "content": user_input_query})
    with st.chat_message("user"):
        st.markdown(user_input_query)

    # Proses dan dapatkan jawaban dari asisten (chatbot)
    if faiss_index is None or not text_chunks or query_embed_model is None or llm_model is None:
        assistant_response = "Maaf, sistem chatbot belum siap sepenuhnya. Beberapa komponen gagal dimuat."
    else:
        # 1. Embed query pengguna
        query_vector = get_query_embedding_vector(user_input_query)
        
        if query_vector is not None:
            # 2. Retrieve chunks relevan
            relevant_chunks = retrieve_from_faiss(query_vector, k=3) # Ambil 3 chunks teratas
            
            # (Opsional) Tampilkan konteks yang digunakan untuk debug Anda
            # if relevant_chunks:
            #     with st.expander("Konteks yang Digunakan (Debug)"):
            #         st.json(relevant_chunks)
            
            # 3. Generate jawaban LLM dengan RAG
            assistant_response = generate_llm_response(user_input_query, relevant_chunks)
        else:
            assistant_response = "Maaf, terjadi masalah saat memproses pertanyaan Anda (gagal membuat embedding query)."

    # Tambahkan respons asisten ke histori dan tampilkan
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)