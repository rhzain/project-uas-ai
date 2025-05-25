import streamlit as st
import json
import faiss
import numpy as np
import os
from dotenv import load_dotenv
import google.generativeai as genai

# # Jika menggunakan Sentence Transformers untuk embedding query
# from sentence_transformers import SentenceTransformer

load_dotenv() # Muat variabel dari .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY tidak ditemukan. Harap atur di file .env atau sebagai environment variable.")
    st.stop() # Hentikan eksekusi jika API key tidak ada
genai.configure(api_key=GEMINI_API_KEY)

# Pilih model Gemini yang sesuai untuk text generation
# Contoh: 'gemini-1.5-flash-latest' atau 'gemini-1.0-pro-latest'
try:
    llm_model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
    st.success("Model LLM Gemini berhasil dimuat.")
except Exception as e:
    st.error(f"Gagal memuat model LLM Gemini: {e}")
    st.stop()
    
@st.cache_resource # Cache resource ini agar tidak reload terus-menerus
def load_rag_resources(index_path, chunks_path, embedding_model_name_for_query=None):
    try:
        index = faiss.read_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            text_chunks_list = json.load(f)

        query_embedding_model = None
        # if embedding_model_name_for_query: # Jika pakai Sentence Transformers untuk query
        #     query_embedding_model = SentenceTransformer(embedding_model_name_for_query)

        # Jika Anda pakai Gemini untuk embedding query, Anda tidak perlu memuat model ST di sini.
        # Fungsi embedding query Gemini akan dipanggil langsung.

        st.success(f"FAISS index dimuat ({index.ntotal} vektor). Text chunks dimuat ({len(text_chunks_list)} chunk).")
        return index, text_chunks_list, query_embedding_model
    except FileNotFoundError:
        st.error(f"Error: Salah satu file ({index_path} atau {chunks_path}) tidak ditemukan.")
        return None, None, None
    except Exception as e:
        st.error(f"Gagal memuat resources RAG: {e}")
        return None, None, None

# Tentukan path ke file Anda
FAISS_INDEX_PATH = "sisop_faiss.index" # GANTI DENGAN NAMA FILE INDEX ANDA
TEXT_CHUNKS_PATH = "sisop_chunks.json"    # GANTI DENGAN NAMA FILE CHUNKS ANDA

# Tentukan nama model embedding yang DIGUNAKAN SAAT MEMBUAT INDEX FAISS jika Anda pakai Sentence Transformers.
# Jika Anda pakai Gemini untuk embedding saat buat index, biarkan ini None.
EMBEDDING_MODEL_FOR_QUERY = 'all-MiniLM-L6-v2' # Ganti jika Anda pakai model lain, atau None jika query juga pakai Gemini Embedding

# Muat resources
faiss_index, text_chunks, query_st_model = load_rag_resources(FAISS_INDEX_PATH, TEXT_CHUNKS_PATH, EMBEDDING_MODEL_FOR_QUERY)

def get_query_embedding(query_text):
    if query_st_model: # Jika menggunakan Sentence Transformers (model sudah dimuat)
        return query_st_model.encode([query_text])[0]
    else: # Jika menggunakan Gemini untuk embedding query
        try:
            # Ganti dengan model embedding Gemini yang Anda gunakan untuk membuat index, misal 'models/text-embedding-004'
            response = genai.embed_content(
                model="models/text-embedding-004", # PASTIKAN MODEL INI SAMA DENGAN YANG DIPAKAI UNTUK DOKUMEN
                content=query_text,
                task_type="RETRIEVAL_QUERY" # Atau "SEMANTIC_SIMILARITY"
            )
            return np.array(response['embedding'])
        except Exception as e:
            st.error(f"Error saat membuat embedding query dengan Gemini: {e}")
            return None
        
def retrieve_relevant_chunks(query_embedding, k=3): # k = jumlah chunk teratas
    # Debugging Awal: Cek kondisi faiss_index dan query_embedding
    if faiss_index is None:
        st.warning("Peringatan Debug: faiss_index belum dimuat (None). Retrieval tidak bisa dilanjutkan.")
        print("DEBUG: faiss_index is None in retrieve_relevant_chunks")
        return []
    if not text_chunks: # Pastikan text_chunks juga ada
        st.warning("Peringatan Debug: text_chunks kosong atau belum dimuat. Retrieval tidak bisa dilanjutkan.")
        print("DEBUG: text_chunks is empty or None in retrieve_relevant_chunks")
        return []
    if query_embedding is None:
        st.warning("Peringatan Debug: query_embedding adalah None. Retrieval tidak bisa dilanjutkan.")
        print("DEBUG: query_embedding is None in retrieve_relevant_chunks")
        return []

    # Informasi Debug Tambahan sebelum pencarian FAISS
    print(f"DEBUG: retrieve_relevant_chunks - faiss_index.ntotal = {faiss_index.ntotal}")
    print(f"DEBUG: retrieve_relevant_chunks - faiss_index.d (dimensi index) = {faiss_index.d}")
    
    query_embedding_np = np.array(query_embedding) # Konversi ke NumPy array jika belum
    print(f"DEBUG: retrieve_relevant_chunks - query_embedding shape = {query_embedding_np.shape}")
    print(f"DEBUG: retrieve_relevant_chunks - query_embedding dtype = {query_embedding_np.dtype}")
    
    try:
        # Pastikan query_embedding_np adalah 2D dan tipenya float32 untuk FAISS
        if query_embedding_np.ndim == 1:
            query_embedding_2d = np.array([query_embedding_np]).astype('float32')
        elif query_embedding_np.ndim == 2 and query_embedding_np.shape[0] == 1:
            query_embedding_2d = query_embedding_np.astype('float32')
        else:
            st.error(f"Error Debug: query_embedding memiliki shape yang tidak diharapkan: {query_embedding_np.shape}")
            print(f"ERROR_DEBUG: query_embedding has unexpected shape: {query_embedding_np.shape}")
            return []

        print(f"DEBUG: retrieve_relevant_chunks - query_embedding_2d shape for FAISS = {query_embedding_2d.shape}")
        print(f"DEBUG: retrieve_relevant_chunks - query_embedding_2d dtype for FAISS = {query_embedding_2d.dtype}")

        # Periksa konsistensi dimensi antara index dan query
        if faiss_index.d != query_embedding_2d.shape[1]:
            st.error(f"Error Kritis: Dimensi tidak cocok! Dimensi index FAISS: {faiss_index.d}, Dimensi query: {query_embedding_2d.shape[1]}")
            print(f"CRITICAL_ERROR_DEBUG: Dimension mismatch! Index: {faiss_index.d}, Query: {query_embedding_2d.shape[1]}")
            return []

        distances, indices = faiss_index.search(query_embedding_2d, k)
        
        print(f"DEBUG: retrieve_relevant_chunks - FAISS search distances = {distances}")
        print(f"DEBUG: retrieve_relevant_chunks - FAISS search indices = {indices}")

        if indices.size == 0 or (indices.ndim > 1 and indices[0].size == 0) : # Jika tidak ada index yang ditemukan
            print("DEBUG: retrieve_relevant_chunks - FAISS search tidak mengembalikan index.")
            return []
            
        relevant_chunks_texts = []
        valid_indices_count = 0
        # indices[0] karena kita hanya mencari untuk satu query_embedding_2d
        for i in indices[0]: 
            if i == -1: # FAISS mengembalikan -1 jika tidak ada tetangga yang ditemukan untuk posisi itu
                print(f"DEBUG: retrieve_relevant_chunks - FAISS mengembalikan index -1, dilewati.")
                continue
            if 0 <= i < len(text_chunks):
                relevant_chunks_texts.append(text_chunks[i])
                valid_indices_count +=1
            else:
                print(f"DEBUG: retrieve_relevant_chunks - Index {i} dari FAISS di luar jangkauan untuk text_chunks (panjang: {len(text_chunks)}). Dilewati.")
        
        print(f"DEBUG: retrieve_relevant_chunks - Jumlah chunk valid yang ditemukan: {valid_indices_count}")
        if not relevant_chunks_texts:
            print("DEBUG: retrieve_relevant_chunks - Tidak ada chunk relevan yang ditemukan setelah memfilter index.")
            
        return relevant_chunks_texts
        
    except Exception as e:
        # Logging error yang lebih detail ke konsol dan UI Streamlit
        error_type = type(e).__name__
        error_message = str(e)
        st.error(f"Error saat melakukan retrieval dari FAISS (Internal): [{error_type}] {error_message}")
        print(f"--- EXCEPTION DETAILS in retrieve_relevant_chunks ---")
        print(f"Type: {error_type}")
        print(f"Message: {error_message}")
        import traceback
        traceback.print_exc() # Ini akan mencetak traceback lengkap ke konsol tempat Streamlit berjalan
        print(f"--- END EXCEPTION DETAILS ---")
        return []
    
def generate_rag_answer(query, context_chunks):
    if not context_chunks:
        # Opsi: Langsung jawab dengan LLM tanpa konteks, atau beri tahu tidak ada info
        # return llm_model.generate_content(f"Jawab pertanyaan berikut: {query}").text
        return "Maaf, saya tidak menemukan informasi yang relevan dalam materi yang tersedia untuk menjawab pertanyaan Anda."

    context_str = "\n\n---\n\n".join(context_chunks)

    prompt = f"""Anda adalah chatbot edukasi AI yang membantu mahasiswa."""
    
st.title("🤖 Chatbot Edukasi AI (dengan RAG)")
st.caption("Ditenagai oleh Gemini dan FAISS")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Tanyakan sesuatu tentang materi..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if faiss_index is None or not text_chunks:
        response = "Maaf, sistem RAG belum siap. Pastikan file index dan chunks sudah dimuat dengan benar."
    else:
        # 1. Embed query
        query_embedding = get_query_embedding(prompt)

        if query_embedding is not None:
            # 2. Retrieve context
            relevant_context = retrieve_relevant_chunks(query_embedding, k=3) # Ambil 3 chunk teratas

            # 3. Generate answer
            response = generate_rag_answer(prompt, relevant_context)

            # (Opsional) Tampilkan konteks yang digunakan untuk debugging
            if relevant_context and st.secrets.get("SHOW_DEBUG_CONTEXT", False): # Tambahkan SHOW_DEBUG_CONTEXT=true di secrets.toml jika ingin lihat
                with st.expander("Konteks yang Digunakan (Debug)"):
                    st.json(relevant_context)
        else:
            response = "Maaf, terjadi masalah saat memproses pertanyaan Anda (gagal membuat embedding)."

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})