import streamlit as st
import json
import faiss
import numpy as np
import os
import re
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import time

# --- 0. Konfigurasi Awal & Pemuatan Variabel Lingkungan ---
st.set_page_config(page_title="Platform Edukasi AI", layout="wide", initial_sidebar_state="expanded")
load_dotenv() 

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Untuk deployment Streamlit Cloud, Anda bisa menggunakan st.secrets
if not GEMINI_API_KEY and hasattr(st, 'secrets') and "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

if not GEMINI_API_KEY:
    st.error("Variabel lingkungan GEMINI_API_KEY tidak ditemukan. Harap atur di file .env atau sebagai environment variable sistem/Streamlit secret.")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Konfigurasi Gemini API berhasil.")
except Exception as e:
    st.error(f"Gagal mengkonfigurasi Gemini API: {e}")
    st.stop()

# --- Variabel Global & Path Konfigurasi ---
AVAILABLE_COURSES = {
    "Sistem Operasi": {
        "id": "sistem_operasi",
        "base_dir": os.path.join("dataset", "SistemOperasi"), # Path relatif dari root aplikasi
        "outline_file": "outline_operating_systems.txt",
        "default_embedding_model": 'all-MiniLM-L6-v2',
        "finetuned_embedding_model_dir": "finetuned_embedding_model_sistem_operasi",
        "faiss_index_finetuned": "vector_store_finetuned.index",
        "chunks_json_finetuned": "processed_chunks_metadata_finetuned.json",
        "faiss_index_base": "vector_store_base.index",
        "chunks_json_base": "processed_chunks_metadata_base.json"
    }
    # Tambahkan course lain di sini jika ada
    # "Nama Course Lain": { "id": "id_unik_lain", ... }
}
LLM_MODEL_NAME = "gemini-1.5-flash-latest" # Model LLM untuk RAG dan generasi soal

# --- Fungsi Helper untuk Membaca File Teks ---
def read_text_file_content(filepath):
    """Membaca konten dari file teks."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di '{filepath}'.")
        return f"Error: File materi tidak ditemukan di '{filepath}'."
    except Exception as e:
        print(f"Error saat membaca file '{filepath}': {e}")
        return f"Error saat memuat materi: {e}"

# --- 1. Fungsi Pemuatan Resources (Disesuaikan untuk Course Tertentu) ---
@st.cache_resource # Cache resource antar sesi untuk course yang sama
def load_course_resources(course_id):
    """Memuat semua resource yang dibutuhkan untuk course yang dipilih."""
    if course_id not in [c_info["id"] for c_info in AVAILABLE_COURSES.values()]:
        st.error(f"Course dengan ID '{course_id}' tidak ditemukan.")
        return None
    
    # Dapatkan nama tampilan dan konfigurasi course berdasarkan ID
    course_name_display = next((c_name for c_name, c_info in AVAILABLE_COURSES.items() if c_info["id"] == course_id), None)
    if not course_name_display:
        st.error(f"Tidak dapat menemukan nama tampilan untuk course ID '{course_id}'.")
        return None
    course_config = AVAILABLE_COURSES[course_name_display]
    
    base_dir = course_config["base_dir"]

    # Tentukan path model embedding dan artefak RAG yang akan dimuat
    path_to_finetuned_model_dir = os.path.join(base_dir, course_config["finetuned_embedding_model_dir"])
    path_to_faiss_finetuned = os.path.join(base_dir, course_config["faiss_index_finetuned"])
    path_to_chunks_finetuned = os.path.join(base_dir, course_config["chunks_json_finetuned"])

    path_to_faiss_base = os.path.join(base_dir, course_config["faiss_index_base"])
    path_to_chunks_base = os.path.join(base_dir, course_config["chunks_json_base"])
    
    embedding_model_to_load = course_config["default_embedding_model"]
    faiss_index_to_load = path_to_faiss_base
    chunks_json_to_load = path_to_chunks_base
    embedding_model_display_name = f"Default ({course_config['default_embedding_model']})"

    # Prioritaskan model fine-tuned jika ada dan artefaknya lengkap
    if os.path.isdir(path_to_finetuned_model_dir) and \
       os.path.exists(path_to_faiss_finetuned) and \
       os.path.exists(path_to_chunks_finetuned):
        embedding_model_to_load = path_to_finetuned_model_dir
        faiss_index_to_load = path_to_faiss_finetuned
        chunks_json_to_load = path_to_chunks_finetuned
        embedding_model_display_name = "Fine-tuned (Lokal)"
        print(f"Akan memuat model embedding fine-tuned untuk course '{course_name_display}'.")
    else:
        print(f"Model embedding fine-tuned atau artefak RAG-nya tidak ditemukan untuk course '{course_name_display}'. Menggunakan default.")
        # Pastikan file default ada
        if not (os.path.exists(path_to_faiss_base) and os.path.exists(path_to_chunks_base)):
            st.error(f"File RAG dasar (FAISS/JSON) juga tidak ditemukan untuk model default course '{course_name_display}'. Harap jalankan skrip persiapan data.")
            return None

    outline_filepath = os.path.join(base_dir, course_config["outline_file"])

    print(f"Memulai pemuatan resources untuk course: {course_name_display}")
    # ... (sisa print path bisa ditambahkan jika perlu untuk debugging) ...

    resources = {
        "faiss_index": None, "text_chunks_with_metadata": [], "parsed_outline": [],
        "query_embedding_model": None, "llm_model": None, 
        "embedding_model_name_loaded": embedding_model_display_name,
        "course_name_display": course_name_display, 
        "course_id": course_id,
        "base_dir": base_dir # Simpan base_dir untuk akses file materi
    }

    all_paths_exist = True
    if not os.path.exists(faiss_index_to_load): st.error(f"File FAISS index TIDAK DITEMUKAN: {os.path.abspath(faiss_index_to_load)}"); all_paths_exist = False
    if not os.path.exists(chunks_json_to_load): st.error(f"File text chunks JSON TIDAK DITEMUKAN: {os.path.abspath(chunks_json_to_load)}"); all_paths_exist = False
    if not os.path.exists(outline_filepath): st.error(f"File outline TIDAK DITEMUKAN: {os.path.abspath(outline_filepath)}"); all_paths_exist = False

    if not all_paths_exist:
        st.warning("Satu atau lebih file data penting tidak ditemukan. Aplikasi mungkin tidak berfungsi dengan benar.")
        return resources # Kembalikan dict dengan beberapa nilai None

    try:
        resources["faiss_index"] = faiss.read_index(faiss_index_to_load)
        with open(chunks_json_to_load, "r", encoding="utf-8") as f:
            resources["text_chunks_with_metadata"] = json.load(f)

        parsed_outline_data = []
        with open(outline_filepath, 'r', encoding='utf-8') as f_outline: content = f_outline.read()
        
        # Ambil nama matakuliah dari baris pertama file outline jika ada
        if content.strip().startswith("MATAKULIAH:"):
            try:
                course_title_from_outline = content.splitlines()[0].split(":",1)[1].strip()
                resources["course_name_display"] = course_title_from_outline # Update dengan nama dari file
            except Exception: pass
        
        pertemuan_blocks = re.split(r'\nPERTEMUAN:', '\n' + content.split('PERTEMUAN:', 1)[-1] if 'PERTEMUAN:' in content else '')
        for block in pertemuan_blocks:
            if not block.strip(): continue
            current_pertemuan = {}
            lines = block.strip().splitlines()
            if lines:
                id_match_from_line_start = re.match(r'^\s*(\d+)', lines[0])
                if id_match_from_line_start: current_pertemuan['id'] = int(id_match_from_line_start.group(1))
                else: continue 

                for line_idx, line_content in enumerate(lines):
                    if ":" in line_content:
                        key_value_match = re.match(r'^\s*([A-Z_]+)\s*:\s*(.*)', line_content, re.IGNORECASE)
                        if key_value_match:
                            key_clean = key_value_match.group(1).strip().lower().replace(" ", "_")
                            value_clean = key_value_match.group(2).strip()
                            current_pertemuan[key_clean] = value_clean
            
            if 'id' in current_pertemuan and 'judul' in current_pertemuan:
                parsed_outline_data.append(current_pertemuan)
            else:
                 print(f"Peringatan: Gagal mem-parsing ID atau Judul untuk blok: {lines[:2] if lines else 'Blok Kosong'}")

        resources["parsed_outline"] = parsed_outline_data
        print(f"Outline mata kuliah dimuat ({len(resources['parsed_outline'])} pertemuan).")
        
        resources["query_embedding_model"] = SentenceTransformer(embedding_model_to_load)
        print(f"Model embedding query '{embedding_model_to_load}' berhasil dimuat.")

        resources["llm_model"] = genai.GenerativeModel(model_name=LLM_MODEL_NAME)
        print(f"Model LLM Gemini '{LLM_MODEL_NAME}' berhasil dimuat.")
        
        st.success(f"Semua resources untuk course '{resources['course_name_display']}' berhasil dimuat!")
        return resources
        
    except Exception as e:
        st.error(f"Terjadi kesalahan fatal saat memuat resources: {e}")
        import traceback
        traceback.print_exc()
        # Kembalikan dict dengan nilai None untuk resource yang gagal, tapi jaga info course
        failed_resources = {key: None for key in resources.keys() if key not in ["course_name_display", "course_id", "base_dir", "embedding_model_name_loaded"]}
        failed_resources.update({
            "course_name_display": resources.get("course_name_display", course_name_display),
            "course_id": resources.get("course_id", course_id),
            "base_dir": resources.get("base_dir", base_dir),
            "embedding_model_name_loaded": resources.get("embedding_model_name_loaded", embedding_model_display_name)
        })
        return failed_resources


# --- 2. Fungsi-Fungsi Inti untuk RAG dan Kuis ---
def get_embedding_for_query(user_query_text, query_model):
    if query_model is None: 
        st.warning("Model embedding query belum siap untuk membuat embedding.")
        return None
    try:
        return query_model.encode([user_query_text])[0]
    except Exception as e:
        print(f"Error embedding query: {e}")
        st.error(f"Error saat membuat embedding untuk query: {e}")
        return None

def search_relevant_chunks(query_embedding_vector, faiss_idx, text_chunks_meta, current_pertemuan_id=None, top_k=3): # Kurangi top_k default untuk RAG
    if faiss_idx is None or query_embedding_vector is None or not text_chunks_meta:
        st.warning("Komponen RAG (FAISS/chunks/query embedding) belum siap untuk pencarian.")
        return []
    try:
        query_np_array = np.array([query_embedding_vector]).astype('float32')
        if faiss_idx.d != query_np_array.shape[1]:
            st.error(f"Dimensi embedding query ({query_np_array.shape[1]}) tidak cocok dengan index FAISS ({faiss_idx.d}).")
            return []
        
        # Ambil lebih banyak jika perlu filter, terutama jika current_pertemuan_id tidak None
        # Namun, untuk RAG, 3-5 chunk biasanya cukup.
        num_to_search = top_k 
        if faiss_idx.ntotal == 0:
            print("FAISS index kosong.")
            return []
        num_to_search = min(num_to_search, faiss_idx.ntotal)


        distances, global_indices = faiss_idx.search(query_np_array, num_to_search)
        retrieved_chunks_texts = []
        for i in global_indices[0]:
            if i != -1 and 0 <= i < len(text_chunks_meta):
                chunk_data = text_chunks_meta[i]
                # Filter berdasarkan ID pertemuan JIKA current_pertemuan_id diberikan
                if current_pertemuan_id is None or str(chunk_data.get("pertemuan_id")) == str(current_pertemuan_id):
                    retrieved_chunks_texts.append(chunk_data["chunk_text"]) 
                    # Tidak perlu break di sini, biarkan mengambil hingga num_to_search
        return retrieved_chunks_texts[:top_k] # Pastikan hanya mengembalikan top_k setelah filter
    except Exception as e:
        print(f"Error saat search_relevant_chunks: {e}")
        st.error(f"Error saat melakukan pencarian di FAISS: {e}")
        return []

def get_rag_answer_from_llm(user_query, context_chunks, llm_model):
    if llm_model is None: 
        st.warning("Model LLM tidak siap untuk menjawab.")
        return "Error: Model LLM tidak siap."
    
    prompt_to_send = ""
    if not context_chunks:
        prompt_to_send = f"Jawab pertanyaan berikut berdasarkan pengetahuan umum Anda, karena tidak ada konteks materi spesifik yang ditemukan: \"{user_query}\""
        print("INFO: Menjawab tanpa konteks RAG karena tidak ada chunk relevan ditemukan.")
    else:
        context_string = "\n\n---\n\n".join(context_chunks)
        prompt_to_send = f"""Anda adalah asisten AI edukasi yang cerdas dan membantu.
Berdasarkan KONTEKS MATERI di bawah ini, jawablah PERTANYAAN MAHASISWA dengan jelas dan akurat.
Fokuskan jawaban Anda HANYA pada informasi yang ada dalam KONTEKS MATERI.
Jika informasi tidak ada dalam konteks, katakan bahwa Anda tidak dapat menemukannya dalam materi yang disediakan.
Hindari membuat asumsi di luar konteks.

KONTEKS MATERI:
---
{context_string}
---

PERTANYAAN MAHASISWA:
"{user_query}"

JAWABAN ANDA (jelas dan ringkas berdasarkan konteks):
"""
    try:
        # Konfigurasi safety settings untuk mengurangi blocking jika terlalu ketat
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        response = llm_model.generate_content(prompt_to_send, safety_settings=safety_settings)

        if response.parts: return "".join(part.text for part in response.parts)
        elif hasattr(response, 'text') and response.text: return response.text
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
            print(f"Respons LLM diblokir: {response.prompt_feedback.block_reason_message}")
            return f"Tidak dapat menghasilkan jawaban karena konten diblokir. Alasan: {response.prompt_feedback.block_reason_message}. Coba ubah pertanyaan Anda."
        else:
            print(f"Respons LLM tidak memiliki 'text' atau 'parts': {response}")
            return "Maaf, format respons dari LLM tidak dikenali atau kosong."
            
    except Exception as e:
        print(f"Error saat get_rag_answer_from_llm: {e}")
        st.error(f"Maaf, terjadi kesalahan saat mencoba menghasilkan jawaban: {e}")
        return "Maaf, terjadi kesalahan internal saat mencoba menghasilkan jawaban."


def generate_mcq_from_llm(pertemuan_id, num_questions, llm_model, text_chunks_meta):
    if llm_model is None: 
        st.warning("Model LLM tidak siap untuk generasi soal."); return []
    
    relevant_chunks_for_quiz = [
        chunk["chunk_text"] for chunk in text_chunks_meta 
        if str(chunk.get("pertemuan_id")) == str(pertemuan_id)
    ]
    if not relevant_chunks_for_quiz:
        st.warning(f"Tidak ada materi chunk yang ditemukan untuk pertemuan ID {pertemuan_id} untuk membuat soal."); return []
    
    # Ambil sampel chunk yang lebih representatif, jangan terlalu panjang
    sample_context_for_quiz = "\n\n---\n\n".join(relevant_chunks_for_quiz[:min(len(relevant_chunks_for_quiz), 5)]) # Batasi jumlah chunk
    # Batasi panjang total konteks
    if len(sample_context_for_quiz) > 8000: # Perkiraan batas token untuk prompt
        sample_context_for_quiz = sample_context_for_quiz[:8000] + "\n... (materi dipotong)"


    prompt_quiz_generation = f"""Anda adalah seorang ahli pembuat soal ujian pilihan ganda berdasarkan materi yang diberikan.
Berdasarkan potongan materi kuliah berikut:
---
{sample_context_for_quiz}
---
Tolong buatkan saya {num_questions} soal pilihan ganda yang menguji pemahaman mahasiswa mengenai konsep-konsep utama dalam materi di atas.
Setiap soal HARUS memiliki:
1. "pertanyaan": Pertanyaan yang jelas dan tidak ambigu.
2. "opsi": Sebuah dictionary berisi EMPAT opsi jawaban (kunci: "A", "B", "C", "D", nilainya adalah teks opsi yang ringkas dan jelas).
3. "jawaban_benar": Kunci dari opsi yang benar (misalnya, "B").
4. "pembahasan": Penjelasan singkat (1-2 kalimat) mengapa jawaban tersebut benar dan mengapa opsi lain salah, berdasarkan materi.
5. "topik_terkait": Topik atau kata kunci spesifik dari materi yang diuji oleh soal ini (maksimal 3 kata, contoh: "Manajemen Memori", "Algoritma FCFS", "Definisi OS").

Format output HARUS berupa list dari JSON object yang valid, seperti ini:
[
  {{
    "pertanyaan": "Apa fungsi utama dari kernel dalam sistem operasi?",
    "opsi": {{ "A": "Mengelola antarmuka pengguna grafis", "B": "Menyediakan layanan inti sistem operasi", "C": "Menjalankan aplikasi pengguna secara langsung", "D": "Melakukan kompilasi kode program" }},
    "jawaban_benar": "B",
    "pembahasan": "Kernel adalah inti dari sistem operasi yang menyediakan layanan fundamental seperti manajemen proses, memori, dan perangkat keras. Opsi lain bukan fungsi utama kernel.",
    "topik_terkait": "Kernel OS"
  }}
  // ... soal lainnya ...
]
Pastikan outputnya adalah JSON list yang valid dan tidak ada teks tambahan di luar list JSON tersebut. Jangan gunakan markdown dalam nilai JSON. Opsi jawaban harus berbeda signifikan.
"""
    print(f"DEBUG: Prompt untuk generasi soal (konteks {len(sample_context_for_quiz)} char)...")
    try:
        safety_settings_quiz = [ # Mungkin perlu lebih permisif untuk generasi soal
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        response = llm_model.generate_content(prompt_quiz_generation, safety_settings=safety_settings_quiz)
        response_text = "".join(part.text for part in response.parts) if response.parts else (response.text if hasattr(response, 'text') else "")
        
        # Mencari blok JSON yang valid dalam respons
        json_match = re.search(r'\[\s*(\{[\s\S]*?\}(?:\s*,\s*\{[\s\S]*?\})*)\s*\]', response_text, re.DOTALL | re.MULTILINE)

        if json_match:
            json_str = json_match.group(0) 
            try:
                questions = json.loads(json_str)
                valid_questions = []
                for q_idx, q in enumerate(questions):
                    if isinstance(q, dict) and \
                       all(key in q for key in ["pertanyaan", "opsi", "jawaban_benar", "pembahasan", "topik_terkait"]) and \
                       isinstance(q["opsi"], dict) and len(q["opsi"]) == 4 and \
                       q["jawaban_benar"] in q["opsi"]:
                        valid_questions.append(q)
                    else:
                        print(f"DEBUG Soal tidak valid dari LLM (idx {q_idx}): {q}")
                        st.warning(f"Soal ke-{q_idx+1} dari LLM memiliki format tidak lengkap/valid dan akan dilewati.")
                
                if not valid_questions and questions:
                     st.error("Semua soal yang dihasilkan LLM memiliki format tidak valid setelah divalidasi.")
                     print(f"Soal yang diparsing tapi tidak valid: {questions}")
                     return []
                elif not questions and response_text.strip(): # Ada teks tapi tidak bisa diparsing sebagai JSON list
                     st.error("Tidak ada soal yang dapat diparsing sebagai JSON list dari respons LLM.")
                     print(f"Respons mentah LLM (gagal parse JSON list): {response_text}")
                     return []

                print(f"Berhasil mem-parsing dan memvalidasi {len(valid_questions)} soal dari LLM.")
                return valid_questions
            except json.JSONDecodeError as je:
                st.error(f"Gagal mem-parsing JSON soal dari LLM: {je}"); print(f"JSON Decode Error. String yang dicoba parse: {json_str}")
                return []
        else:
            st.error("Tidak menemukan format JSON list yang valid dalam respons LLM untuk soal.")
            print(f"Tidak ada JSON list valid ditemukan dalam: {response_text}")
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                 st.warning(f"Generasi soal mungkin diblokir: {response.prompt_feedback.block_reason_message}")
            return []

    except Exception as e:
        st.error(f"Error saat men-generate soal dari LLM: {e}"); print(f"Error detail saat generate_mcq_from_llm: {e}")
        import traceback; traceback.print_exc()
        return []

# --- 3. Inisialisasi Session State ---
# Navigasi & Resources
if "current_view" not in st.session_state: st.session_state.current_view = "course_selection"
if "selected_course_id" not in st.session_state: st.session_state.selected_course_id = None
if "course_resources" not in st.session_state: st.session_state.course_resources = None

# State Pertemuan
if "current_pertemuan_id" not in st.session_state: st.session_state.current_pertemuan_id = None
if "current_pertemuan_judul" not in st.session_state: st.session_state.current_pertemuan_judul = None
if "current_pertemuan_deskripsi" not in st.session_state: st.session_state.current_pertemuan_deskripsi = ""
if "current_pertemuan_full_materi" not in st.session_state: st.session_state.current_pertemuan_full_materi = ""

# State Kuis
if "quiz_mode" not in st.session_state: st.session_state.quiz_mode = None # None, "generating", "ongoing", "results"
if "quiz_questions" not in st.session_state: st.session_state.quiz_questions = []
if "current_question_index" not in st.session_state: st.session_state.current_question_index = 0
if "user_answers" not in st.session_state: st.session_state.user_answers = {}
if "quiz_submitted" not in st.session_state: st.session_state.quiz_submitted = False
if "quiz_score_details" not in st.session_state: st.session_state.quiz_score_details = {}

# State Chat
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "auto_send_prompt_topic" not in st.session_state: st.session_state.auto_send_prompt_topic = None


# --- Fungsi untuk mereset state saat ganti course atau pertemuan ---
def reset_meeting_specific_state():
    st.session_state.chat_history = []
    st.session_state.quiz_mode = None
    st.session_state.quiz_questions = []
    st.session_state.current_question_index = 0
    st.session_state.user_answers = {}
    st.session_state.quiz_submitted = False
    st.session_state.quiz_score_details = {}
    st.session_state.auto_send_prompt_topic = None

def reset_course_specific_state():
    st.session_state.current_pertemuan_id = None
    st.session_state.current_pertemuan_judul = None
    st.session_state.current_pertemuan_deskripsi = ""
    st.session_state.current_pertemuan_full_materi = ""
    reset_meeting_specific_state()


# --- Navigasi Utama Aplikasi ---
if st.session_state.current_view == "course_selection":
    st.title("Selamat Datang di Platform Edukasi AI 🎓")
    st.subheader("Pilih Mata Kuliah:")

    # Menggunakan kolom untuk tata letak tombol course yang lebih baik jika ada banyak
    course_names = list(AVAILABLE_COURSES.keys())
    num_cols = min(len(course_names), 3) # Maksimal 3 kolom
    cols = st.columns(num_cols) if num_cols > 0 else [st] # Fallback ke st jika tidak ada course

    for idx, course_name in enumerate(course_names):
        course_info = AVAILABLE_COURSES[course_name]
        current_col = cols[idx % num_cols]
        with current_col:
            if st.button(f"Mulai Belajar: {course_name}", key=f"course_{course_info['id']}", use_container_width=True, type="primary"):
                st.session_state.selected_course_id = course_info["id"]
                st.session_state.current_view = "meeting_view"
                reset_course_specific_state() # Reset semua state terkait course dan pertemuan
                st.session_state.course_resources = None # Paksa muat ulang resource
                st.rerun()
    
    st.markdown("---")
    st.info("Aplikasi ini adalah prototipe. Untuk saat ini, hanya mata kuliah 'Sistem Operasi' yang memiliki data lengkap.")

elif st.session_state.current_view == "meeting_view" and st.session_state.selected_course_id:
    # Muat resources untuk course yang dipilih jika belum ada atau course berubah
    if st.session_state.course_resources is None or \
       st.session_state.course_resources.get("course_id") != st.session_state.selected_course_id:
        with st.spinner(f"Memuat data mata kuliah... Ini mungkin memakan waktu beberapa saat."):
            st.session_state.course_resources = load_course_resources(st.session_state.selected_course_id)
    
    app_res = st.session_state.course_resources # Alias untuk kemudahan akses
    
    # Validasi apakah semua resource penting berhasil dimuat
    if not app_res or not all(app_res.get(key) for key in ["faiss_index", "text_chunks_with_metadata", "query_embedding_model", "llm_model", "parsed_outline", "base_dir"]):
        st.error("Gagal memuat semua resource yang dibutuhkan untuk mata kuliah ini. Silakan coba lagi atau periksa file data Anda.")
        if st.button("Kembali ke Pilihan Mata Kuliah"):
            st.session_state.current_view = "course_selection"
            st.session_state.selected_course_id = None
            st.session_state.course_resources = None 
            st.cache_resource.clear() # Clear cache juga
            st.rerun()
        st.stop() # Hentikan eksekusi jika resource kritis tidak ada

    # --- Sidebar untuk Navigasi Pertemuan ---
    with st.sidebar:
        st.header(f"📖 {app_res['course_name_display']}")
        st.caption(f"Model Embedding: {app_res['embedding_model_name_loaded']}")
        st.divider()
        st.subheader("Navigasi Pertemuan:")
        if not app_res["parsed_outline"]:
            st.warning("Outline mata kuliah tidak berhasil dimuat atau kosong.")
        else:
            for pertemuan in app_res["parsed_outline"]:
                pertemuan_id_outline = str(pertemuan.get("id")) 
                judul = pertemuan.get("judul", f"Pertemuan {pertemuan_id_outline}")
                deskripsi_p = pertemuan.get("deskripsi", "Tidak ada deskripsi.") # Ambil deskripsi untuk tooltip atau info
                file_materi_rel = pertemuan.get("file_materi")

                btn_type = "primary" if str(st.session_state.current_pertemuan_id) == pertemuan_id_outline else "secondary"
                if st.button(judul, key=f"pertemuan_{app_res['course_id']}_{pertemuan_id_outline}", use_container_width=True, type=btn_type, help=deskripsi_p):
                    if str(st.session_state.current_pertemuan_id) != pertemuan_id_outline:
                        st.session_state.current_pertemuan_id = pertemuan_id_outline
                        st.session_state.current_pertemuan_judul = judul
                        st.session_state.current_pertemuan_deskripsi = deskripsi_p # Simpan deskripsi juga
                        
                        # Muat materi lengkap untuk pertemuan yang dipilih
                        full_materi_content = "Materi tidak dapat dimuat atau file tidak ditemukan."
                        if file_materi_rel:
                            material_file_abs_path = os.path.join(app_res["base_dir"], file_materi_rel)
                            full_materi_content = read_text_file_content(material_file_abs_path)
                        st.session_state.current_pertemuan_full_materi = full_materi_content
                        
                        reset_meeting_specific_state() # Reset kuis dan chat
                        st.rerun()
        st.divider()
        if st.button("⬅️ Kembali ke Pilihan Mata Kuliah", use_container_width=True):
            st.session_state.current_view = "course_selection"
            st.session_state.selected_course_id = None
            reset_course_specific_state() # Reset semua state course
            st.session_state.course_resources = None 
            st.cache_resource.clear() 
            st.rerun()
        
        if st.button("🔄 Reset Course Ini", type="secondary", use_container_width=True, help="Muat ulang resource untuk course ini dan reset state pertemuan."):
            st.cache_resource.clear() 
            st.session_state.course_resources = None 
            reset_course_specific_state()
            st.rerun()


    # --- Tampilan Utama Aplikasi untuk Pertemuan ---
    if st.session_state.current_pertemuan_id is None:
        st.title(f"Selamat Datang di Mata Kuliah: {app_res['course_name_display']}")
        st.write("Silakan pilih salah satu pertemuan dari menu navigasi di sebelah kiri untuk memulai pembelajaran.")
        st.markdown("---")
        st.subheader("Ikhtisar Semua Pertemuan:")
        if not app_res["parsed_outline"]:
            st.info("Belum ada detail pertemuan untuk mata kuliah ini.")
        else:
            for p_data in app_res["parsed_outline"]:
                pid = str(p_data.get("id"))
                judul_p = p_data.get("judul", f"Pertemuan {pid}")
                desk_p = p_data.get("deskripsi", "Tidak ada deskripsi.")
                with st.expander(f"**Pertemuan {pid}: {judul_p}**"):
                    st.markdown(f"_{desk_p}_")
    else:
        st.title(f"📍 Pertemuan {st.session_state.current_pertemuan_id}: {st.session_state.current_pertemuan_judul}")
        
        # Gunakan tab untuk memisahkan konten
        tab_materi, tab_kuis, tab_chat = st.tabs(["📜 Materi Lengkap", "📝 Kuis Pemahaman", "💬 Tanya Jawab AI"])

        with tab_materi:
            st.subheader("Materi Lengkap Pertemuan")
            # Menggunakan st.markdown jika materi Anda memiliki format markdown.
            # Jika teks biasa, st.text_area atau st.code bisa lebih sesuai.
            # Untuk teks panjang, text_area dengan disabled=True dan height baik.
            if st.session_state.current_pertemuan_full_materi.startswith("Error:"):
                st.error(st.session_state.current_pertemuan_full_materi)
            else:
                st.markdown(st.session_state.current_pertemuan_full_materi, unsafe_allow_html=True) # Hati-hati dengan unsafe_allow_html jika sumber tidak terpercaya

        with tab_kuis:
            st.subheader("Uji Pemahaman Anda")
            if st.session_state.quiz_mode is None:
                if st.button("💡 Mulai Kuis untuk Pertemuan Ini", type="primary", use_container_width=True, key=f"start_quiz_{st.session_state.current_pertemuan_id}"):
                    st.session_state.quiz_mode = "generating"
                    st.rerun()
            
            if st.session_state.quiz_mode == "generating":
                with st.spinner("Sedang membuat soal kuis untuk Anda... Ini mungkin memerlukan beberapa saat. ⏳"):
                    # Pastikan semua argumen diteruskan dengan benar
                    st.session_state.quiz_questions = generate_mcq_from_llm(
                        st.session_state.current_pertemuan_id, 
                        num_questions=3, # Jumlah soal yang diinginkan
                        llm_model=app_res["llm_model"],
                        text_chunks_meta=app_res["text_chunks_with_metadata"]
                    )
                    if st.session_state.quiz_questions: # Jika ada soal yang valid
                        st.session_state.current_question_index = 0
                        st.session_state.user_answers = {}
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_score_details = {"total_soal": len(st.session_state.quiz_questions)}
                        st.session_state.quiz_mode = "ongoing"
                    else: 
                        st.error("Gagal membuat soal kuis. Silakan coba lagi atau materi mungkin kurang memadai.")
                        st.session_state.quiz_mode = None # Kembali ke state netral
                    st.rerun()
            
            elif st.session_state.quiz_mode == "ongoing":
                q_idx = st.session_state.current_question_index
                # Cek apakah soal masih ada dan index valid
                if not st.session_state.quiz_questions or q_idx >= len(st.session_state.quiz_questions) :
                    st.session_state.quiz_mode = "results"
                    st.session_state.quiz_submitted = True # Anggap disubmit jika kehabisan soal
                    st.rerun()
                
                question_data = st.session_state.quiz_questions[q_idx]
                
                st.markdown(f"**Soal {q_idx + 1} dari {len(st.session_state.quiz_questions)}:**")
                st.markdown(f"##### {question_data.get('pertanyaan', 'Pertanyaan tidak tersedia.')}")
                
                options = question_data.get("opsi", {})
                if not isinstance(options, dict) or len(options) < 2 : # Minimal 2 opsi, idealnya 4
                    st.error("Format opsi soal tidak valid. Kuis tidak dapat dilanjutkan.")
                    if st.button("Kembali", key="quiz_option_error_back"): st.session_state.quiz_mode = None; st.rerun()
                    st.stop()

                option_keys = sorted(list(options.keys())) 

                current_answer_for_q = st.session_state.user_answers.get(q_idx)
                default_radio_idx = None
                if current_answer_for_q and current_answer_for_q in option_keys:
                    default_radio_idx = option_keys.index(current_answer_for_q)

                selected_key = st.radio(
                    "Pilih jawaban Anda:",
                    options=option_keys,
                    format_func=lambda key_opt: f"{key_opt}. {options[key_opt]}",
                    key=f"quiz_q_{app_res['course_id']}_{st.session_state.current_pertemuan_id}_{q_idx}",
                    index=default_radio_idx,
                    horizontal=False # Tampilkan vertikal agar lebih mudah dibaca
                )
                if selected_key: 
                    st.session_state.user_answers[q_idx] = selected_key

                # Navigasi Kuis
                quiz_nav_cols = st.columns([1, 1, 1.5, 1]) # Sesuaikan rasio kolom
                with quiz_nav_cols[0]:
                    if st.button("⬅️ Sebelumnya", disabled=(q_idx == 0), use_container_width=True):
                        st.session_state.current_question_index -= 1
                        st.rerun()
                with quiz_nav_cols[1]:
                    if q_idx < len(st.session_state.quiz_questions) - 1:
                        if st.button("Berikutnya ➡️", disabled=(st.session_state.user_answers.get(q_idx) is None), use_container_width=True):
                            st.session_state.current_question_index += 1
                            st.rerun()
                with quiz_nav_cols[2]:
                     if q_idx == len(st.session_state.quiz_questions) - 1: 
                        if st.button("🏁 Selesai & Lihat Hasil", type="primary", disabled=(st.session_state.user_answers.get(q_idx) is None), use_container_width=True):
                            st.session_state.quiz_submitted = True
                            st.session_state.quiz_mode = "results"
                            st.rerun()
                with quiz_nav_cols[3]:
                    if st.button("❌ Batalkan Kuis", use_container_width=True, type="secondary"):
                        st.session_state.quiz_mode = None
                        st.session_state.quiz_questions = [] 
                        st.rerun()

            elif st.session_state.quiz_mode == "results":
                st.subheader("📊 Hasil Uji Pemahaman Anda")
                if not st.session_state.quiz_submitted or not st.session_state.quiz_questions:
                    st.warning("Belum ada kuis yang diselesaikan atau soal tidak tersedia untuk ditampilkan hasilnya.")
                else:
                    benar, salah = 0, 0
                    topik_salah_dict = {}
                    total_soal_quiz = len(st.session_state.quiz_questions)
                    
                    for idx, q_data in enumerate(st.session_state.quiz_questions):
                        user_ans = st.session_state.user_answers.get(idx)
                        correct_ans_key = q_data.get("jawaban_benar")
                        if user_ans == correct_ans_key:
                            benar += 1
                        else:
                            salah += 1
                            topik = q_data.get("topik_terkait", "Topik Umum") # Default jika tidak ada
                            topik_salah_dict[topik] = topik_salah_dict.get(topik, 0) + 1
                    
                    st.session_state.quiz_score_details = {"benar": benar, "salah": salah, "total_soal": total_soal_quiz, "topik_salah": topik_salah_dict}
                    
                    if total_soal_quiz > 0:
                        persentase_benar = (benar / total_soal_quiz) * 100
                        st.metric(label="Skor Anda", value=f"{persentase_benar:.2f}%", delta=f"{benar} benar dari {total_soal_quiz} soal")

                        st.markdown("---")
                        st.markdown("#### Detail Jawaban dan Pembahasan:")
                        for idx, q_data in enumerate(st.session_state.quiz_questions):
                            exp_title = f"Soal {idx+1}: {q_data.get('pertanyaan','N/A')}"
                            user_ans_key = st.session_state.user_answers.get(idx)
                            is_wrong = (user_ans_key != q_data.get("jawaban_benar"))
                            
                            with st.expander(exp_title, expanded=is_wrong): # Otomatis expand jika salah
                                user_ans_text = q_data.get("opsi",{}).get(user_ans_key, "Tidak Dijawab")
                                correct_ans_key = q_data.get("jawaban_benar")
                                correct_ans_text = q_data.get("opsi",{}).get(correct_ans_key, "N/A")

                                st.markdown(f"**Jawaban Anda:** {user_ans_key or ''}. {user_ans_text} {'✔️' if not is_wrong else '❌'}")
                                st.markdown(f"**Jawaban Benar:** {correct_ans_key or ''}. {correct_ans_text}")
                                st.info(f"**Pembahasan:** {q_data.get('pembahasan', 'Tidak ada pembahasan untuk soal ini.')}")
                                st.caption(f"Topik Terkait: {q_data.get('topik_terkait', 'Tidak diketahui')}")

                                if is_wrong:
                                    topik_soal_salah = q_data.get('topik_terkait', 'materi soal ini') # Fallback jika topik kosong
                                    if st.button(f"💬 Tanya AI tentang: '{topik_soal_salah}'", key=f"ask_results_topic_{idx}"):
                                        st.session_state.auto_send_prompt_topic = topik_soal_salah
                                        st.session_state.quiz_mode = "chat_from_quiz_topic" # State untuk trigger di tab chat
                                        st.rerun() # Akan pindah ke tab chat dan memproses auto_send_prompt_topic
                    else:
                        st.warning("Tidak ada soal yang dijawab untuk ditampilkan hasilnya.")

                if st.button("Ulangi Kuis", key="retake_quiz_button", use_container_width=True):
                    st.session_state.quiz_mode = "generating"
                    st.rerun()
                if st.button("Kembali ke Materi Pertemuan", key="back_to_meeting_from_results", use_container_width=True):
                    st.session_state.quiz_mode = None
                    st.session_state.quiz_questions = [] 
                    st.rerun()

        with tab_chat:
            st.subheader("Tanya Jawab Materi dengan Asisten AI")
            
            # Jika ada trigger dari hasil kuis untuk bertanya topik
            if st.session_state.quiz_mode == "chat_from_quiz_topic" and st.session_state.auto_send_prompt_topic:
                auto_query = f"Tolong jelaskan lebih detail mengenai topik '{st.session_state.auto_send_prompt_topic}' dari Pertemuan {st.session_state.current_pertemuan_id} ({st.session_state.current_pertemuan_judul})."
                st.session_state.chat_history.append({"role": "user", "content": auto_query})
                
                query_vector = get_embedding_for_query(auto_query, app_res["query_embedding_model"])
                assistant_response = "Maaf, ada masalah saat memproses embedding pertanyaan Anda."
                if query_vector is not None:
                    relevant_chunks = search_relevant_chunks(query_vector, app_res["faiss_index"], app_res["text_chunks_with_metadata"], current_pertemuan_id=st.session_state.current_pertemuan_id)
                    assistant_response = get_rag_answer_from_llm(auto_query, relevant_chunks, app_res["llm_model"])
                
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                st.session_state.auto_send_prompt_topic = None # Hapus setelah diproses
                st.session_state.quiz_mode = None # Kembali ke mode chat normal
                st.rerun() # Rerun untuk menampilkan chat baru

            # Tampilkan riwayat chat
            for message_entry in st.session_state.chat_history:
                with st.chat_message(message_entry["role"]):
                    st.markdown(message_entry["content"])

            # Input chat dari pengguna
            if user_chat_input := st.chat_input(f"Tanyakan sesuatu tentang Pertemuan {st.session_state.current_pertemuan_id}..."):
                st.session_state.chat_history.append({"role": "user", "content": user_chat_input})
                
                query_vector = get_embedding_for_query(user_chat_input, app_res["query_embedding_model"])
                assistant_response = "Maaf, ada masalah saat memproses embedding pertanyaan Anda."
                if query_vector is not None:
                    relevant_chunks = search_relevant_chunks(query_vector, app_res["faiss_index"], app_res["text_chunks_with_metadata"], current_pertemuan_id=st.session_state.current_pertemuan_id)
                    assistant_response = get_rag_answer_from_llm(user_chat_input, relevant_chunks, app_res["llm_model"])
                
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                st.rerun()

else: # Fallback jika state current_view tidak terduga
    st.session_state.current_view = "course_selection"
    st.rerun()