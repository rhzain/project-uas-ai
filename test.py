from datetime import datetime
import streamlit as st

st.set_page_config(page_title="Chatbot Edukasi AI", layout="wide")

import json
import faiss
import numpy as np
import os
import re
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from feedback_system import FeedbackCollector
# from gemini_fine_tuning import GeminiFinetuner
# from embedding_fine_tuning import EmbeddingFineTuner
# from rl_training import RLTrainer, create_rl_batches

# --- 0. Konfigurasi Awal & Pemuatan Variabel Lingkungan ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Variabel lingkungan GEMINI_API_KEY tidak ditemukan. Harap atur API Key Anda.")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Konfigurasi Gemini API berhasil.")
except Exception as e:
    st.error(f"Gagal mengkonfigurasi Gemini API: {e}")
    st.stop()

# --- Variabel Global & Path Konfigurasi ---
# Sesuaikan dengan nama model embedding yang Anda gunakan di prepare_data_for_rag.py
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

CURRENT_WORKING_DIRECTORY = os.getcwd()
print(f"Direktori Kerja Saat Ini (CWD): {CURRENT_WORKING_DIRECTORY}")

BASE_DATA_DIR = os.path.join("dataset", "SistemOperasi") 

print(f"BASE_DATA_DIR diatur ke (relatif terhadap CWD atau absolut): {BASE_DATA_DIR}")
print(f"Path absolut yang akan digunakan untuk BASE_DATA_DIR: {os.path.abspath(BASE_DATA_DIR)}")


FAISS_INDEX_FILEPATH = os.path.join(BASE_DATA_DIR, "vector_store.index")
TEXT_CHUNKS_FILEPATH = os.path.join(BASE_DATA_DIR, "processed_chunks_with_metadata.json")
OUTLINE_FILEPATH = os.path.join(BASE_DATA_DIR, "outline_operating_systems.txt")

# Model LLM Gemini yang akan digunakan
LLM_MODEL_NAME = "gemini-1.5-flash-latest"

# --- 1. Pemuatan Resources Utama (Menggunakan Cache Streamlit) ---
@st.cache_resource
def load_all_application_resources():
    """Memuat semua resource yang dibutuhkan aplikasi."""
    print("Memulai pemuatan resources aplikasi...")
    print(f"  Mencoba memuat FAISS index dari: {os.path.abspath(FAISS_INDEX_FILEPATH)}")
    print(f"  Mencoba memuat Text Chunks dari: {os.path.abspath(TEXT_CHUNKS_FILEPATH)}")
    print(f"  Mencoba memuat Outline dari: {os.path.abspath(OUTLINE_FILEPATH)}")
    
    resources = {
        "faiss_index": None,
        "text_chunks_with_metadata": [],
        "parsed_outline": [],
        "query_embedding_model": None,
        "llm_model": None
    }
    all_paths_exist = True
    if not os.path.exists(FAISS_INDEX_FILEPATH):
        st.error(f"File FAISS index TIDAK DITEMUKAN di: {os.path.abspath(FAISS_INDEX_FILEPATH)}")
        all_paths_exist = False
    if not os.path.exists(TEXT_CHUNKS_FILEPATH):
        st.error(f"File text chunks JSON TIDAK DITEMUKAN di: {os.path.abspath(TEXT_CHUNKS_FILEPATH)}")
        all_paths_exist = False
    if not os.path.exists(OUTLINE_FILEPATH):
        st.error(f"File outline mata kuliah TIDAK DITEMUKAN di: {os.path.abspath(OUTLINE_FILEPATH)}")
        all_paths_exist = False

    if not all_paths_exist:
        st.warning("Satu atau lebih file data penting tidak ditemukan. Aplikasi mungkin tidak berfungsi dengan benar. Harap periksa path di atas dan pastikan file ada.")
        return {key: None for key in resources}

    try:
        # Muat FAISS Index
        resources["faiss_index"] = faiss.read_index(FAISS_INDEX_FILEPATH)
        print(f"FAISS index dimuat ({resources['faiss_index'].ntotal} vektor).")

        # Muat Text Chunks dengan Metadata
        with open(TEXT_CHUNKS_FILEPATH, "r", encoding="utf-8") as f:
            resources["text_chunks_with_metadata"] = json.load(f)
        print(f"Text chunks dimuat ({len(resources['text_chunks_with_metadata'])} chunk).")

        # Muat dan Parse Outline Mata Kuliah
        parsed_outline_data = []
        with open(OUTLINE_FILEPATH, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if content.strip().startswith("MATAKULIAH:"):
            try:
                st.session_state.nama_matakuliah = content.splitlines()[0].split(":",1)[1].strip()
            except Exception: 
                 pass


        pertemuan_blocks = re.split(r'\nPERTEMUAN:', '\n' + content.split('PERTEMUAN:', 1)[-1] if 'PERTEMUAN:' in content else '')
        for block in pertemuan_blocks:
            if not block.strip(): continue
            current_pertemuan = {}
            lines = block.strip().splitlines()
            if lines:
                # Coba parse ID dari awal baris pertama blok
                id_match_from_line_start = re.match(r'^\s*(\d+)', lines[0])
                if id_match_from_line_start:
                    current_pertemuan['id'] = int(id_match_from_line_start.group(1))
                
                # Proses sisa baris untuk KEY: VALUE
                for line_idx, line in enumerate(lines):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key_clean = key.strip().lower().replace(" ", "_")
                        value_clean = value.strip()
                        
                        if key_clean == "pertemuan" and 'id' not in current_pertemuan:
                             id_match_val = re.match(r'^\s*(\d+)', value_clean)
                             if id_match_val:
                                 current_pertemuan['id'] = int(id_match_val.group(1))
                        elif key_clean not in current_pertemuan or key_clean == 'judul': # Ambil judul dari baris pertama jika ada
                            if key_clean == 'judul' and line_idx == 0 and 'id' in current_pertemuan:
                                # Khusus untuk kasus "ID JUDUL: Isi Judul"
                                current_pertemuan[key_clean] = value_clean.split(":",1)[-1].strip() if ":" in value_clean and value_clean.startswith(str(current_pertemuan['id'])) else value_clean
                            elif key_clean == 'judul' and lines[0].strip().startswith(str(current_pertemuan.get('id','')) + " " + key.strip()) :
                                current_pertemuan[key_clean] = value_clean
                            elif key_clean != 'pertemuan':
                                current_pertemuan[key_clean] = value_clean
            
            if 'id' in current_pertemuan and 'judul' in current_pertemuan:
                parsed_outline_data.append(current_pertemuan)
            else:
                print(f"Peringatan: Gagal mem-parsing ID atau Judul untuk blok: {lines[:2]}")

        resources["parsed_outline"] = parsed_outline_data
        print(f"Outline mata kuliah dimuat ({len(resources['parsed_outline'])} pertemuan).")
        
        # Muat Model Embedding untuk Query
        print(f"Memuat model embedding untuk query: {EMBEDDING_MODEL_NAME}...")
        resources["query_embedding_model"] = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Model embedding query berhasil dimuat.")

        # Muat Model LLM Gemini
        print(f"Memuat model LLM Gemini: {LLM_MODEL_NAME}...")
        resources["llm_model"] = genai.GenerativeModel(model_name=LLM_MODEL_NAME)
        print("Model LLM Gemini berhasil dimuat.")
        
        st.success("Semua resources aplikasi berhasil dimuat!")
        return resources
        
    except FileNotFoundError as fnf_error:
        st.error(f"Kesalahan File Tidak Ditemukan saat memuat resources: {fnf_error}")
        print(f"Detail FileNotFoundError saat load_all_application_resources: {fnf_error}")
        import traceback
        traceback.print_exc()
        return {key: None for key in resources} # Kembalikan None untuk semua resource
    except Exception as e:
        st.error(f"Terjadi kesalahan fatal saat memuat resources: {e}")
        print(f"Detail error umum saat load_all_application_resources: {e}")
        import traceback
        traceback.print_exc()
        return {key: None for key in resources}


# Panggil fungsi untuk memuat resources
app_resources = load_all_application_resources()
faiss_search_index = app_resources.get("faiss_index") # Gunakan .get() untuk keamanan
loaded_text_chunks_with_metadata = app_resources.get("text_chunks_with_metadata", [])
parsed_outline = app_resources.get("parsed_outline", [])
query_embedding_model = app_resources.get("query_embedding_model")
llm_chat_model = app_resources.get("llm_model")


# --- 2. Fungsi-Fungsi Inti untuk RAG dan Kuis ---

def get_embedding_for_query(user_query_text):
    if query_embedding_model is None: 
        st.warning("Model embedding query belum siap.")
        return None
    try:
        return query_embedding_model.encode([user_query_text])[0]
    except Exception as e:
        print(f"Error embedding query: {e}")
        st.error(f"Error saat membuat embedding untuk query: {e}")
        return None

def search_relevant_chunks(query_embedding_vector, current_pertemuan_id=None, top_k=5):
    if faiss_search_index is None or query_embedding_vector is None or not loaded_text_chunks_with_metadata:
        st.warning("Komponen RAG (FAISS/chunks/query embedding) belum siap untuk pencarian.")
        return []
    try:
        query_np_array = np.array([query_embedding_vector]).astype('float32')
        if faiss_search_index.d != query_np_array.shape[1]:
            st.error(f"Dimensi embedding query ({query_np_array.shape[1]}) tidak cocok dengan index FAISS ({faiss_search_index.d}).")
            return []

        num_to_search = top_k * 5 if current_pertemuan_id is not None else top_k 
        num_to_search = min(num_to_search, faiss_search_index.ntotal)

        distances, global_indices = faiss_search_index.search(query_np_array, num_to_search)

        retrieved_chunks_texts = []
        for i in global_indices[0]:
            if i != -1 and 0 <= i < len(loaded_text_chunks_with_metadata):
                chunk_data = loaded_text_chunks_with_metadata[i]
                if current_pertemuan_id is None or chunk_data.get("pertemuan_id") == current_pertemuan_id:
                    retrieved_chunks_texts.append(chunk_data["chunk_text"]) 
                    if len(retrieved_chunks_texts) == top_k:
                        break 
        
        # Store context for feedback
        st.session_state.last_context = "\n\n---\n\n".join(retrieved_chunks_texts)
        
        return retrieved_chunks_texts
    except Exception as e:
        print(f"Error saat search_relevant_chunks: {e}")
        st.error(f"Error saat melakukan pencarian di FAISS: {e}")
        return []

def get_rag_answer_from_llm(user_query, context_chunks):
    if llm_chat_model is None: 
        st.warning("Model LLM tidak siap.")
        return "Error: Model LLM tidak siap."
    
    prompt_to_send = ""
    if not context_chunks:
        prompt_to_send = f"Jawab pertanyaan berikut berdasarkan pengetahuan umum Anda: \"{user_query}\""
        print("INFO: Menjawab tanpa konteks RAG karena tidak ada chunk relevan ditemukan.")
    else:
        context_string = "\n\n---\n\n".join(context_chunks)
        prompt_to_send = f"""Anda adalah asisten AI edukasi yang cerdas dan membantu.
Berdasarkan KONTEKS MATERI di bawah ini, jawablah PERTANYAAN MAHASISWA dengan jelas dan akurat.
Fokuskan jawaban Anda HANYA pada informasi yang ada dalam KONTEKS MATERI.
Jika informasi tidak ada dalam konteks, katakan bahwa Anda tidak dapat menemukannya dalam materi yang disediakan.

KONTEKS MATERI:
---
{context_string}
---

PERTANYAAN MAHASISWA:
"{user_query}"

JAWABAN ANDA:
"""
    try:
        response = llm_chat_model.generate_content(prompt_to_send)
        if response.parts:
            return "".join(part.text for part in response.parts)
        elif hasattr(response, 'text') and response.text:
            return response.text
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            feedback_info = response.prompt_feedback
            print(f"Feedback dari LLM untuk prompt: {feedback_info}")
            return f"Tidak dapat menghasilkan jawaban. Feedback: {feedback_info}"
        else:
            print(f"Respons LLM tidak memiliki 'text' atau 'parts': {response}")
            return "Maaf, format respons dari LLM tidak dikenali."
            
    except Exception as e:
        print(f"Error saat get_rag_answer_from_llm: {e}")
        st.error(f"Maaf, terjadi kesalahan saat mencoba menghasilkan jawaban: {e}")
        return "Maaf, terjadi kesalahan internal saat mencoba menghasilkan jawaban."


def generate_mcq_from_llm(pertemuan_id, num_questions=3):
    if llm_chat_model is None: 
        st.warning("Model LLM tidak siap untuk generasi soal.")
        return []
    
    relevant_chunks_for_quiz = [
        chunk["chunk_text"] for chunk in loaded_text_chunks_with_metadata 
        if chunk.get("pertemuan_id") == pertemuan_id
    ]
    if not relevant_chunks_for_quiz:
        st.warning(f"Tidak ada materi chunk yang ditemukan untuk pertemuan ID {pertemuan_id} untuk membuat soal.")
        return []
    
    sample_context_for_quiz = "\n\n---\n\n".join(relevant_chunks_for_quiz[:min(len(relevant_chunks_for_quiz), 5)])
    
    prompt_quiz_generation = f"""Anda adalah seorang ahli pembuat soal ujian.
Berdasarkan potongan materi kuliah berikut:
---
{sample_context_for_quiz}
---
Tolong buatkan saya {num_questions} soal pilihan ganda yang menguji pemahaman mahasiswa mengenai konsep-konsep utama dalam materi di atas.
Untuk setiap soal, sertakan:
1. "pertanyaan": Pertanyaan yang jelas.
2. "opsi": Sebuah dictionary berisi empat opsi jawaban (kunci: "A", "B", "C", "D").
3. "jawaban_benar": Kunci dari opsi yang benar (misalnya, "B").
4. "penjelasan_jawaban": Penjelasan singkat mengapa jawaban tersebut benar dan opsi lain salah.
5. "topik_terkait": Topik atau sub-bagian spesifik dari materi yang diuji oleh soal ini.

Format output HARUS berupa list dari JSON object yang valid, seperti ini:
[
  {{
    "pertanyaan": "Contoh pertanyaan 1...",
    "opsi": {{ "A": "Opsi A1", "B": "Opsi B1", "C": "Opsi C1", "D": "Opsi D1" }},
    "jawaban_benar": "A",
    "penjelasan_jawaban": "Penjelasan untuk soal 1...",
    "topik_terkait": "Topik terkait soal 1"
  }},
  {{
    "pertanyaan": "Contoh pertanyaan 2...",
    "opsi": {{ "A": "Opsi A2", "B": "Opsi B2", "C": "Opsi C2", "D": "Opsi D2" }},
    "jawaban_benar": "C",
    "penjelasan_jawaban": "Penjelasan untuk soal 2...",
    "topik_terkait": "Topik terkait soal 2"
  }}
]
Pastikan outputnya adalah JSON list yang valid dan tidak ada teks tambahan di luar list JSON tersebut.
"""
    print(f"DEBUG: Prompt untuk generasi soal (sebagian):\n{prompt_quiz_generation[:300]}...")
    try:
        response = llm_chat_model.generate_content(prompt_quiz_generation)
        response_text = ""
        if response.parts:
            response_text = "".join(part.text for part in response.parts)
        elif hasattr(response, 'text') and response.text:
            response_text = response.text
        
        print(f"DEBUG: Respons mentah dari LLM untuk generasi soal:\n{response_text}")
        
        # Mencari blok JSON yang valid dalam respons
        # Ini lebih toleran terhadap teks tambahan sebelum atau sesudah JSON
        json_match = re.search(r'\[\s*(\{[\s\S]*?\}(?:\s*,\s*\{[\s\S]*?\})*)\s*\]', response_text, re.DOTALL)

        if json_match:
            json_str = json_match.group(0) # Ambil seluruh match [ ... ]
            try:
                questions = json.loads(json_str)
                # Validasi sederhana struktur soal
                if isinstance(questions, list) and all(isinstance(q, dict) and "pertanyaan" in q and "opsi" in q and "jawaban_benar" in q for q in questions):
                    print(f"Berhasil mem-parsing {len(questions)} soal dari LLM.")
                    return questions
                else:
                    st.error("Format JSON soal dari LLM tidak sesuai ekspektasi setelah parsing.")
                    print(f"Format JSON soal tidak sesuai. Parsed: {questions}")
                    return []
            except json.JSONDecodeError as je:
                st.error(f"Gagal mem-parsing JSON soal dari LLM: {je}")
                print(f"JSON Decode Error. String yang dicoba parse: {json_str}")
                return []
        else:
            st.error("Tidak menemukan format JSON list yang valid dalam respons LLM untuk soal.")
            print(f"Tidak ada JSON list valid ditemukan dalam: {response_text}")
            # Cek jika ada feedback blocking
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                 if response.prompt_feedback.block_reason:
                    st.warning(f"Generasi soal mungkin diblokir: {response.prompt_feedback.block_reason_message}")
            return []

    except Exception as e:
        st.error(f"Error saat men-generate soal dari LLM: {e}")
        print(f"Error detail saat generate_mcq_from_llm: {e}")
        import traceback
        traceback.print_exc()
        return []

# --- 3. Inisialisasi Session State ---
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "current_pertemuan_id" not in st.session_state: st.session_state.current_pertemuan_id = None
if "current_pertemuan_judul" not in st.session_state: st.session_state.current_pertemuan_judul = None
if "quiz_questions" not in st.session_state: st.session_state.quiz_questions = []
if "current_question_index" not in st.session_state: st.session_state.current_question_index = 0
if "user_answers" not in st.session_state: st.session_state.user_answers = {}
if "quiz_mode" not in st.session_state: st.session_state.quiz_mode = None
if "quiz_score" not in st.session_state: st.session_state.quiz_score = {"benar": 0, "salah": 0, "total_soal":0, "topik_salah": {}}
if "pemahaman_mahasiswa" not in st.session_state: st.session_state.pemahaman_mahasiswa = {}
if "nama_matakuliah" not in st.session_state: st.session_state.nama_matakuliah = "Mata Kuliah Anda" # Default
if "last_processed_query" not in st.session_state: st.session_state.last_processed_query = None

# --- 4. Feedback System ---
feedback_collector = FeedbackCollector()

def trigger_retraining():
    """Trigger model retraining based on collected feedback"""
    with st.spinner("Retraining models with your feedback..."):
        
        # 1. Fine-tune embedding model
        if len(feedback_collector.feedback_db) >= 10:  # Minimum feedback threshold
            retrain_embedding_model()
        
        # 2. Update LLM with RL
        if len(feedback_collector.feedback_db) >= 20:
            retrain_llm_with_rl()
        
        st.success("Models updated successfully!")

def retrain_embedding_model():
    """Retrain embedding model with user feedback"""
    # Prepare training data from feedback
    training_data = feedback_collector.prepare_embedding_training_data()
    
    # Fine-tune model
    embedding_trainer = EmbeddingFineTuner(EMBEDDING_MODEL_NAME)
    updated_model = embedding_trainer.fine_tune(
        training_data, 
        output_path=f"models/updated_embedding_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Update global model
    global query_embedding_model
    query_embedding_model = updated_model

def retrain_llm_with_rl():
    """Retrain LLM using reinforcement learning"""
    rl_data = feedback_collector.prepare_rl_dataset()
    
    # Initialize RL trainer
    rl_trainer = RLTrainer(LLM_MODEL_NAME)
    
    # Training loop
    for batch in create_rl_batches(rl_data):
        questions = [item['state']['question'] for item in batch]
        contexts = [item['state']['context'] for item in batch]
        answers = [item['action'] for item in batch]
        rewards = [item['reward'] for item in batch]
        
        stats = rl_trainer.train_step(questions, contexts, answers, rewards)
        print(f"RL Training stats: {stats}")

# --- 5. Antarmuka Pengguna (UI) Streamlit ---

# Sidebar untuk Navigasi Pertemuan
with st.sidebar:
    st.header(f"📚 {st.session_state.nama_matakuliah}")
    st.subheader("Daftar Pertemuan:")
    if not parsed_outline:
        st.warning("Outline mata kuliah tidak berhasil dimuat atau kosong.")
    else:
        for pertemuan in parsed_outline:
            pertemuan_id = pertemuan.get("id")
            judul = pertemuan.get("judul", f"Pertemuan {pertemuan_id}")
            if pertemuan_id is None: continue # Lewati jika ID tidak ada

            level_paham = st.session_state.pemahaman_mahasiswa.get(pertemuan_id)
            display_judul = judul
            if level_paham == "Sangat Paham": display_judul += " ✅"
            elif level_paham == "Paham Sebagian": display_judul += " 👍"
            elif level_paham == "Perlu Belajar Lagi": display_judul += " ⚠️"

            if st.button(display_judul, key=f"pertemuan_{pertemuan_id}", use_container_width=True):
                if st.session_state.current_pertemuan_id != pertemuan_id:
                    st.session_state.current_pertemuan_id = pertemuan_id
                    st.session_state.current_pertemuan_judul = judul
                    st.session_state.quiz_mode = None 
                    st.session_state.chat_history = [] 
                    st.session_state.last_processed_query = None
                    st.rerun()

    st.divider()
    if st.sidebar.button("🔄 Reset Aplikasi & Muat Ulang Resources", type="primary", use_container_width=True):
        st.cache_resource.clear()
        keys_to_reset = ["chat_history", "current_pertemuan_id", "current_pertemuan_judul", 
                         "quiz_questions", "current_question_index", "user_answers", 
                         "quiz_mode", "quiz_score", "pemahaman_mahasiswa", "last_processed_query", "nama_matakuliah"]
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
        
    st.divider()
    with st.sidebar.expander("🤖 AI Training Status"):
        stats = feedback_collector.get_feedback_stats()
        
        st.metric("Total Feedback", stats.get('total_feedback', 0))
        if stats.get('avg_rating'):
            st.metric("Average Rating", f"{stats['avg_rating']:.2f}/5")
        
        # Show when models can be retrained
        if stats.get('total_feedback', 0) >= 10:
            st.success("✅ Ready for embedding fine-tuning")
        else:
            needed = 10 - stats.get('total_feedback', 0)
            st.info(f"Need {needed} more feedback for embedding training")
        
        if stats.get('total_feedback', 0) >= 20:
            st.success("✅ Ready for RL training")
        else:
            needed = 20 - stats.get('total_feedback', 0)
            st.info(f"Need {needed} more feedback for RL training")

# Tampilan Utama Aplikasi
if st.session_state.current_pertemuan_id is None:
    st.title("Selamat Datang di Chatbot Edukasi AI!")
    st.write("Silakan pilih pertemuan dari menu di sebelah kiri untuk memulai.")
else:
    st.title(f"📍 Pertemuan {st.session_state.current_pertemuan_id}: {st.session_state.current_pertemuan_judul}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💬 Tanya Jawab Materi Ini", use_container_width=True, disabled=(st.session_state.quiz_mode is not None and st.session_state.quiz_mode != "quiz_results_chat_forward")):
            if st.session_state.quiz_mode != None:
                st.session_state.quiz_mode = None
                st.session_state.chat_history = []
                st.session_state.last_processed_query = None
                st.rerun()
    with col2:
        if st.button("📝 Uji Pemahaman Saya", use_container_width=True, type="primary", disabled=(st.session_state.quiz_mode == "quiz_ongoing")):
            st.session_state.quiz_mode = "quiz_generating"
            st.session_state.chat_history = []
            st.session_state.last_processed_query = None
            st.rerun()
    st.divider()

    if st.session_state.quiz_mode == "quiz_generating":
        with st.spinner("Sedang membuat soal untuk Anda... Mohon tunggu sebentar. ⏳"):
            st.session_state.quiz_questions = generate_mcq_from_llm(st.session_state.current_pertemuan_id, num_questions=3)
            if st.session_state.quiz_questions and isinstance(st.session_state.quiz_questions, list) and len(st.session_state.quiz_questions) > 0 :
                st.session_state.current_question_index = 0
                st.session_state.user_answers = {}
                st.session_state.quiz_score = {"benar": 0, "salah": 0, "total_soal": len(st.session_state.quiz_questions), "topik_salah": {}}
                st.session_state.quiz_mode = "quiz_ongoing"
            else:
                st.error("Gagal membuat soal kuis atau format soal tidak valid. Silakan coba lagi atau pilih pertemuan lain.")
                st.session_state.quiz_mode = None
            st.rerun()
    
    elif st.session_state.quiz_mode == "quiz_ongoing":
        st.subheader("✍️ Kuis Pemahaman")
        if not st.session_state.quiz_questions or st.session_state.current_question_index >= len(st.session_state.quiz_questions):
            st.session_state.quiz_mode = "quiz_results"
            st.rerun()
        else:
            q_idx = st.session_state.current_question_index
            question_data = st.session_state.quiz_questions[q_idx]
            
            st.markdown(f"**Soal {q_idx + 1} dari {len(st.session_state.quiz_questions)}:**")
            st.markdown(f"##### {question_data.get('pertanyaan', 'Pertanyaan tidak tersedia.')}")
            
            options = question_data.get("opsi", {})
            option_items = list(options.items())
            
            if not option_items:
                st.error("Opsi jawaban tidak tersedia untuk soal ini.")
                if st.button("Lanjut ke Soal Berikutnya (jika ada)"):
                    if q_idx < len(st.session_state.quiz_questions) - 1:
                        st.session_state.current_question_index +=1
                    else:
                        st.session_state.quiz_mode = "quiz_results"
                    st.rerun()
                # return # Hentikan render soal ini jika tidak ada opsi

            user_choice_for_current_q = st.session_state.user_answers.get(q_idx)
            
            default_radio_index = None
            if user_choice_for_current_q:
                try:
                    default_radio_index = [item[0] for item in option_items].index(user_choice_for_current_q)
                except ValueError:
                    default_radio_index = None

            selected_option_key = st.radio(
                "Pilih jawaban Anda:",
                options=[item[0] for item in option_items],
                format_func=lambda key: f"{key}. {options.get(key, 'Opsi tidak valid')}",
                key=f"quiz_q_{st.session_state.current_pertemuan_id}_{q_idx}",
                index=default_radio_index
            )
            
            if selected_option_key:
                 st.session_state.user_answers[q_idx] = selected_option_key

            nav_cols = st.columns(3)
            with nav_cols[0]:
                if st.button("⬅️ Soal Sebelumnya", disabled=(q_idx == 0), use_container_width=True):
                    st.session_state.current_question_index -= 1
                    st.rerun()
            with nav_cols[1]:
                if q_idx == len(st.session_state.quiz_questions) - 1:
                    if st.button("🏁 Selesai Kuis & Lihat Hasil", type="primary", use_container_width=True, disabled=(selected_option_key is None and user_choice_for_current_q is None)):
                        if selected_option_key: st.session_state.user_answers[q_idx] = selected_option_key
                        st.session_state.quiz_mode = "quiz_results"
                        st.rerun()
                else:
                    if st.button("Soal Berikutnya ➡️", use_container_width=True, disabled=(selected_option_key is None and user_choice_for_current_q is None)):
                        if selected_option_key: st.session_state.user_answers[q_idx] = selected_option_key
                        st.session_state.current_question_index += 1
                        st.rerun()
            with nav_cols[2]:
                if st.button("❌ Batalkan Kuis", use_container_width=True):
                    st.session_state.quiz_mode = None
                    st.rerun()

    elif st.session_state.quiz_mode == "quiz_results":
        st.subheader("📊 Hasil Uji Pemahaman Anda")
        benar, salah = 0, 0
        topik_salah_dict = {}
        total_soal_quiz = len(st.session_state.quiz_questions)
        st.session_state.quiz_score["total_soal"] = total_soal_quiz


        for idx, q_data in enumerate(st.session_state.quiz_questions):
            user_ans = st.session_state.user_answers.get(idx)
            correct_ans_key = q_data.get("jawaban_benar")
            if user_ans == correct_ans_key:
                benar += 1
            else:
                salah += 1
                topik = q_data.get("topik_terkait", "Umum")
                topik_salah_dict[topik] = topik_salah_dict.get(topik, 0) + 1
        
        st.session_state.quiz_score["benar"] = benar
        st.session_state.quiz_score["salah"] = salah
        st.session_state.quiz_score["topik_salah"] = topik_salah_dict
        
        if total_soal_quiz > 0:
            persentase_benar = (benar / total_soal_quiz) * 100
            st.metric(label="Skor Anda", value=f"{persentase_benar:.2f}%", delta=f"{benar} benar dari {total_soal_quiz} soal")

            level_paham = "Perlu Belajar Lagi"
            if persentase_benar > 80: level_paham = "Sangat Paham"
            elif persentase_benar >= 60: level_paham = "Paham Sebagian"
            
            if level_paham == "Sangat Paham": st.success("🎉 Luar biasa! Pemahaman Anda sangat baik.")
            elif level_paham == "Paham Sebagian": st.info("👍 Bagus! Ada beberapa poin yang bisa ditingkatkan.")
            else: st.warning("⚠️ Anda perlu mempelajari lagi beberapa bagian.")
            
            st.session_state.pemahaman_mahasiswa[st.session_state.current_pertemuan_id] = level_paham

            if topik_salah_dict:
                st.markdown("#### Topik yang Perlu Diperdalam:")
                for topik, count in topik_salah_dict.items():
                    st.markdown(f"- **{topik}** ({count} kesalahan)")
                    if st.button(f"💬 Tanya tentang: {topik}", key=f"learn_{st.session_state.current_pertemuan_id}_{topik.replace(' ','_')}"):
                        st.session_state.chat_history = [] 
                        st.session_state.quiz_mode = "quiz_results_chat_forward" 
                        st.session_state.auto_send_prompt = f"Tolong jelaskan lebih detail mengenai topik '{topik}' dari Pertemuan {st.session_state.current_pertemuan_id} ({st.session_state.current_pertemuan_judul})."
                        st.rerun()
            st.markdown("---")
            
            if st.session_state.chat_history and st.session_state.last_processed_query:
                last_message = st.session_state.chat_history[-1]
                
                if last_message["role"] == "assistant":
                    st.markdown("---")
                    st.subheader("📝 Rate this answer:")
                    
                    feedback_col1, feedback_col2, feedback_col3 = st.columns([3, 1, 1])
                    
                    with feedback_col1:
                        rating = st.selectbox(
                            "How helpful was this answer?",
                            [1, 2, 3, 4, 5],
                            index=2,  # Default to 3 (neutral)
                            format_func=lambda x: f"{x} ⭐ - {['Very Poor', 'Poor', 'Average', 'Good', 'Excellent'][x-1]}",
                            key=f"rating_{len(st.session_state.chat_history)}"
                        )
                        
                        feedback_explanation = st.text_input(
                            "Optional: Explain your rating",
                            placeholder="What could be improved?",
                            key=f"feedback_text_{len(st.session_state.chat_history)}"
                        )
                    
                    with feedback_col2:
                        if st.button("Submit Feedback", key=f"submit_feedback_{len(st.session_state.chat_history)}"):
                            # Collect feedback
                            context_used = st.session_state.get('last_context', '')
                            
                            feedback_collector.collect_answer_feedback(
                                question=st.session_state.last_processed_query,
                                answer=last_message["content"],
                                context=context_used,
                                rating=rating,
                                explanation=feedback_explanation if feedback_explanation else None
                            )
                            
                            st.success("Thank you for your feedback!")
                            
                            # Show feedback stats
                            stats = feedback_collector.get_feedback_stats()
                            st.info(f"Total feedback collected: {stats['total_feedback']}")
                    
                    with feedback_col3:
                        # Show option to retrain models if enough feedback
                        stats = feedback_collector.get_feedback_stats()
                        if stats['total_feedback'] >= 5:  # Minimum threshold
                            if st.button("🔄 Update Models", key=f"retrain_{len(st.session_state.chat_history)}"):
                                trigger_retraining()
            
            st.markdown("#### Detail Jawaban:")
            for idx, q_data in enumerate(st.session_state.quiz_questions):
                with st.expander(f"Soal {idx+1}: {q_data.get('pertanyaan','N/A')}"):
                    user_ans_key = st.session_state.user_answers.get(idx)
                    user_ans_text = q_data.get("opsi",{}).get(user_ans_key, "Tidak Dijawab")
                    correct_ans_key = q_data.get("jawaban_benar")
                    correct_ans_text = q_data.get("opsi",{}).get(correct_ans_key, "N/A")

                    st.markdown(f"**Jawaban Anda:** {user_ans_key or ''}. {user_ans_text}")
                    st.markdown(f"**Jawaban Benar:** {correct_ans_key or ''}. {correct_ans_text}")
                    if user_ans_key == correct_ans_key: st.success("✔️ Benar")
                    else: st.error("❌ Salah")
                    st.markdown(f"**Penjelasan:** {q_data.get('penjelasan_jawaban', 'Tidak ada penjelasan.')}")
                    st.caption(f"Topik Terkait: {q_data.get('topik_terkait', 'Tidak diketahui')}")
        else:
            st.warning("Tidak ada soal yang dijawab untuk ditampilkan hasilnya.")

        if st.button("⬅️ Kembali ke Pilihan Aksi"):
            st.session_state.quiz_mode = None
            st.rerun()

    # Mode Tanya Jawab RAG (default atau setelah forwarding dari hasil kuis)
    else: 
        for message_entry in st.session_state.chat_history:
            with st.chat_message(message_entry["role"]):
                st.markdown(message_entry["content"])

        # Logika untuk auto-send prompt setelah kuis
        if st.session_state.quiz_mode == "quiz_results_chat_forward" and "auto_send_prompt" in st.session_state:
            auto_query = st.session_state.auto_send_prompt
            del st.session_state.auto_send_prompt 
            st.session_state.quiz_mode = None

            st.session_state.chat_history.append({"role": "user", "content": auto_query})
            
            query_vector = get_embedding_for_query(auto_query)
            if query_vector is not None:
                relevant_chunks = search_relevant_chunks(query_vector, current_pertemuan_id=st.session_state.current_pertemuan_id)
                assistant_response = get_rag_answer_from_llm(auto_query, relevant_chunks)
            else:
                assistant_response = "Gagal memproses prompt otomatis (embedding error)."
            
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
            st.session_state.last_processed_query = auto_query
            st.rerun() 

        # Terima input chat dari pengguna
        if user_chat_input := st.chat_input("Ketik pertanyaan Anda di sini..."):
            st.session_state.chat_history.append({"role": "user", "content": user_chat_input})
            
            query_vector = get_embedding_for_query(user_chat_input)
            if query_vector is not None:
                relevant_chunks = search_relevant_chunks(query_vector, current_pertemuan_id=st.session_state.current_pertemuan_id)
                assistant_response = get_rag_answer_from_llm(user_chat_input, relevant_chunks)
            else:
                assistant_response = "Maaf, ada masalah saat memproses embedding pertanyaan Anda."
            
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
            st.session_state.last_processed_query = user_chat_input
            st.rerun() 


if not all(app_resources.values()):
    st.error("Beberapa resource penting gagal dimuat. Aplikasi tidak dapat berjalan. Silakan cek konsol server untuk detail dan coba 'Reset Aplikasi'.")
    st.stop()