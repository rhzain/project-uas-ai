import streamlit as st
import json
import faiss
import numpy as np
import os
import re
import glob
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import torch
from custom_model_integration import get_custom_model_loader

try:
    from reranker import get_reranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

# --- 0. Konfigurasi Awal & Pemuatan Variabel Lingkungan ---

st.set_page_config(
    page_title="Project UAS Artificial Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_dotenv()

# --- Fungsi Penemuan Mata Kuliah Dinamis ---
def discover_available_courses():
    """Enhanced discovery dengan support untuk custom .pth models"""
    courses = {}
    dataset_base = "dataset"
    if not os.path.exists(dataset_base):
        return courses

    for course_dir in os.listdir(dataset_base):
        course_path = os.path.join(dataset_base, course_dir)
        if not os.path.isdir(course_path):
            continue
            
        json_files = glob.glob(os.path.join(course_path, "processed_chunks_metadata_*.json"))
        outline_files = glob.glob(os.path.join(course_path, "outline_*.txt"))
        
        if not json_files:
            continue
            
        course_display_name = course_dir.replace("_", " ").title()
        if outline_files:
            try:
                with open(outline_files[0], 'r', encoding='utf-8') as f:
                    content = f.read()
                    matkul_match = re.search(r'MATAKULIAH:\s*(.*)', content, re.IGNORECASE)
                    if matkul_match:
                        course_display_name = matkul_match.group(1).strip()
            except Exception:
                pass
        
        # Enhanced model detection dengan priority system
        finetuned_model_path = None
        model_type = "📝 Base Model"
        is_custom_pth = False
        
        # Priority 1: SentenceTransformer directory format (existing)
        course_specific_model_path = os.path.join(course_path, f"finetuned_embedding_model_{course_dir.lower()}")
        if os.path.isdir(course_specific_model_path) and os.path.exists(os.path.join(course_specific_model_path, "config.json")):
            finetuned_model_path = course_specific_model_path
            model_type = "🔧 SentenceTransformer Fine-tuned"
        
        # Priority 2: Custom .pth file (NEW - highest priority)
        elif os.path.exists("custom_finetuned_model.pth"):
            finetuned_model_path = "custom_finetuned_model.pth"
            model_type = "🎯 Custom PyTorch Fine-tuned"
            is_custom_pth = True
        
        # Priority 3: Course-specific .pth file  
        elif os.path.exists(os.path.join(course_path, "custom_finetuned_model.pth")):
            finetuned_model_path = os.path.join(course_path, "custom_finetuned_model.pth")
            model_type = f"🎯 Custom {course_dir} Fine-tuned"
            is_custom_pth = True
        
        # File selection logic
        use_finetuned_artifacts = finetuned_model_path is not None
        chunks_file = os.path.join(course_path, "processed_chunks_metadata_finetuned.json") if use_finetuned_artifacts and os.path.exists(os.path.join(course_path, "processed_chunks_metadata_finetuned.json")) else json_files[0]
        faiss_file = os.path.join(course_path, "vector_store_finetuned.index") if use_finetuned_artifacts and os.path.exists(os.path.join(course_path, "vector_store_finetuned.index")) else os.path.join(course_path, "vector_store_base.index")
        
        courses[course_display_name] = {
            "id": course_dir.lower(),
            "base_dir": course_path,
            "chunks_json": chunks_file,
            "outline_file": outline_files[0] if outline_files else None,
            "embedding_model_path": finetuned_model_path or 'all-MiniLM-L6-v2',
            "faiss_index": faiss_file,
            "model_type": model_type,
            "has_finetuned": use_finetuned_artifacts,
            "base_model_name": 'all-MiniLM-L6-v2',
            "is_custom_pth": is_custom_pth  # NEW FLAG
        }
        
    return courses

# --- Konfigurasi API dan Variabel Global ---
AVAILABLE_COURSES = discover_available_courses()
LLM_MODEL_NAME = "gemini-1.5-flash-latest"
device = torch.device("cpu") 

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Variabel lingkungan GEMINI_API_KEY tidak ditemukan."); st.stop()
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Gagal mengkonfigurasi Gemini API: {e}"); st.stop()


# --- 1. Fungsi Pemuatan Resources ---

@st.cache_resource
def load_embedding_model(model_path):
    """Memuat model embedding dari path yang diberikan."""
    try:
        model = SentenceTransformer(model_path, device=device)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model embedding dari '{model_path}': {e}"); return None

@st.cache_resource
def load_course_resources(course_id):
    """Enhanced loading dengan support untuk custom .pth models"""
    course_name = next((name for name, info in AVAILABLE_COURSES.items() if info["id"] == course_id), None)
    if not course_name: 
        return None
    
    config = AVAILABLE_COURSES[course_name]
    
    # Initialize resources
    resources = { 
        "config": config,
        "course_name": course_name,
        "course_id": course_id
    }

    # Enhanced model loading dengan .pth support
    with st.spinner(f"Memuat model embedding ({config['model_type']})..."):
        try:
            embedding_model_path = config["embedding_model_path"]
            
            if config.get("is_custom_pth", False):
                # Load custom .pth model (NEW PATH)
                st.info(f"🎯 Loading custom PyTorch model: {os.path.basename(embedding_model_path)}")
                custom_loader = get_custom_model_loader()
                resources["embedding_model"] = custom_loader.load_custom_pth_model(embedding_model_path)
                
                if resources["embedding_model"] is not None:
                    # Validate custom model
                    if custom_loader.validate_custom_model(resources["embedding_model"]):
                        st.success(f"✅ Custom fine-tuned model berhasil dimuat: {config['model_type']}")
                        
                        # Optional: Show model comparison
                        with st.expander("🔍 Model Comparison Analysis"):
                            base_model = SentenceTransformer('all-MiniLM-L6-v2')
                            comparison = custom_loader.compare_models(
                                base_model, resources["embedding_model"]
                            )
                            if 'error' not in comparison:
                                st.json(comparison)
                            else:
                                st.warning(f"Comparison failed: {comparison['error']}")
                    else:
                        st.warning("⚠️ Model loaded tapi validasi gagal, menggunakan base model")
                        resources["embedding_model"] = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                else:
                    st.warning(f"⚠️ Custom model gagal dimuat, menggunakan base model")
                    resources["embedding_model"] = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                    
            elif embedding_model_path != 'all-MiniLM-L6-v2':  
                # SentenceTransformer directory format (existing path)
                if os.path.isdir(embedding_model_path) and os.path.exists(os.path.join(embedding_model_path, "config.json")):
                    resources["embedding_model"] = SentenceTransformer(embedding_model_path, device=device)
                    st.success(f"✅ SentenceTransformer fine-tuned model: {config['model_type']}")
                else:
                    st.warning(f"⚠️ SentenceTransformer model tidak valid, menggunakan base model")
                    resources["embedding_model"] = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            else:  
                # Base model (existing path)
                resources["embedding_model"] = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                st.info("ℹ️ Menggunakan base model.")
            
            # Set aliases
            resources["retriever_model"] = resources["embedding_model"]
            
        except Exception as e:
            st.error(f"❌ Gagal memuat model embedding: {e}")
            return None
    
    # Validate that embedding model loaded successfully
    if resources["embedding_model"] is None:
        st.error("❌ Model embedding tidak berhasil dimuat")
        return None
    
    # Load chunks data
    try:
        with open(config["chunks_json"], "r", encoding="utf-8") as f: 
            resources["chunks_data"] = json.load(f)
        st.info(f"📚 Berhasil memuat {len(resources['chunks_data'])} chunks materi")
    except Exception as e: 
        st.error(f"❌ Gagal memuat file chunks: {e}")
        return None

    # Load or create FAISS index
    if os.path.exists(config["faiss_index"]):
        with st.spinner("Memuat FAISS index..."):
            try:
                resources["faiss_index"] = faiss.read_index(config["faiss_index"])
                st.success(f"✅ FAISS Index berhasil dimuat ({resources['faiss_index'].ntotal} vectors)")
            except Exception as e:
                st.error(f"❌ Gagal memuat FAISS index: {e}")
                return None
    else:
        st.warning("⚠️ FAISS Index tidak ditemukan. Membuat index baru...")
        passages = [c['chunk_text'] for c in resources["chunks_data"]]
        if passages:
            with st.spinner("🔄 Membuat embedding untuk semua materi..."):
                corpus_embeddings = resources["embedding_model"].encode(passages, convert_to_tensor=True, device=device)
            
            # Create FAISS index
            embedding_dim = corpus_embeddings.shape[1]
            index = faiss.IndexFlatIP(embedding_dim)  # Inner Product for cosine similarity
            index.add(corpus_embeddings.cpu().numpy())
            resources["faiss_index"] = index
            
            # Save the index
            os.makedirs(os.path.dirname(config["faiss_index"]), exist_ok=True)
            faiss.write_index(index, config["faiss_index"])
            st.success(f"✅ FAISS Index baru berhasil dibuat ({index.ntotal} vectors)")
    
    # Load outline (keep existing parsing logic)
    resources["outline"] = []
    if config.get("outline_file") and os.path.exists(config["outline_file"]):
        try:
            with open(config["outline_file"], 'r', encoding='utf-8') as f: 
                content = f.read()
            
            # Parse outline with improved error handling
            for block in re.split(r'\nPERTEMUAN:', content, flags=re.IGNORECASE)[1:]:
                if not block.strip(): 
                    continue
                
                p_data = {}
                
                # Parse ID dari baris pertama
                first_line = block.strip().splitlines()[0] if block.strip().splitlines() else ""
                id_match = re.match(r'^\s*(\d+)', first_line)
                if id_match:
                    p_data['id'] = id_match.group(1)
                
                # Parse key-value pairs dari setiap baris
                for line in block.strip().splitlines():
                    if ":" in line:
                        # Split hanya pada ":" pertama
                        line_parts = line.split(":", 1)
                        if len(line_parts) == 2:
                            key_part = line_parts[0].strip().lower().replace(" ", "_")
                            value_part = line_parts[1].strip()
                            p_data[key_part] = value_part
                
                # Tambahkan ke outline jika memiliki ID dan judul
                if p_data.get('id') and p_data.get('judul'): 
                    resources["outline"].append(p_data)
                    
        except Exception as e:
            st.warning(f"⚠️ Gagal memuat outline: {e}")
            resources["outline"] = []

    # Load LLM model
    try:
        resources["llm_model"] = genai.GenerativeModel(model_name=LLM_MODEL_NAME)
    except Exception as e:
        st.error(f"❌ Gagal memuat LLM model: {e}")
        return None

    st.success(f"🎉 Semua resource untuk '{course_name}' berhasil dimuat!")
    return resources

# --- 2. Fungsi Inti Aplikasi ---

def search_relevant_chunks_with_reranking(query_text, resources, top_k=5, similarity_threshold=0.15, use_reranking=True):
    """Enhanced search dengan reranking dan detailed scoring."""
    
    # Validate resources
    if not resources:
        st.error("Resources tidak tersedia")
        return []
    
    embedding_model = resources.get("embedding_model") or resources.get("retriever_model")
    faiss_index = resources.get("faiss_index")
    chunks_data = resources.get("chunks_data")
    
    if not embedding_model or not faiss_index or not chunks_data:
        st.error("Model embedding, FAISS index, atau chunks data tidak tersedia")
        return []
    
    try:
        # Stage 1: Initial retrieval dengan FAISS
        query_embedding = embedding_model.encode([query_text])
        query_vector = query_embedding[0]
        query_np = np.array([query_vector]).astype('float32')
        
        # Search dengan lebih banyak kandidat untuk reranking
        initial_k = min(top_k * 3, faiss_index.ntotal)  # Ambil 3x lebih banyak untuk reranking
        distances, indices = faiss_index.search(query_np, initial_k)
        
        # Collect initial candidates
        initial_candidates = []
        candidate_metadata = []
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(chunks_data) and dist > similarity_threshold:
                chunk_data = chunks_data[idx]
                initial_candidates.append(chunk_data["chunk_text"])
                candidate_metadata.append({
                    'pertemuan_id': chunk_data.get('pertemuan_id'),
                    'pertemuan_judul': chunk_data.get('pertemuan_judul'),
                    'chunk_id': chunk_data.get('chunk_id'),
                    'faiss_score': float(dist),
                    'faiss_rank': len(initial_candidates)
                })
        
        if not initial_candidates:
            return []
        
        # Stage 2: Reranking (jika tersedia dan diaktifkan)
        if use_reranking and RERANKER_AVAILABLE:
            reranker = get_reranker()
            if reranker:
                # Rerank dengan detailed scores
                reranked_results = reranker.rerank_with_detailed_scores(
                    query_text, 
                    initial_candidates, 
                    candidate_metadata
                )
                
                # Store detailed results untuk UI
                st.session_state.last_search_results = reranked_results[:top_k]
                
                return reranked_results[:top_k]
            else:
                st.warning("⚠️ Reranker tidak tersedia, menggunakan FAISS ranking")
        
        # Fallback: Return FAISS results dengan basic scoring
        basic_results = []
        for i, (chunk, metadata) in enumerate(zip(initial_candidates[:top_k], candidate_metadata[:top_k])):
            result = {
                'text': chunk,
                'rank': i + 1,
                'scores': {
                    'final_score': metadata['faiss_score'],
                    'semantic_similarity': metadata['faiss_score'],
                    'keyword_relevance': 0.5,  # Default value
                    'content_quality': 0.7   # Default value
                },
                'relevance_grade': "🟡 FAISS Similarity",
                'metadata': metadata
            }
            basic_results.append(result)
        
        st.session_state.last_search_results = basic_results
        return basic_results
        
    except Exception as e:
        st.error(f"Error saat mencari chunks: {e}")
        return []

# Update the original search function to use enhanced version
def search_relevant_chunks(query_text, resources, top_k=5, similarity_threshold=0.15):
    """Wrapper function untuk backward compatibility"""
    results = search_relevant_chunks_with_reranking(
        query_text, resources, top_k, similarity_threshold, use_reranking=True
    )
    
    # Return just the text for backward compatibility
    return [result['text'] for result in results] if results else []

def get_rag_answer(query, context_chunks, llm_model):
    """Generate answer HANYA berdasarkan konteks dari dataset, tanpa pengetahuan umum LLM."""
    
    if not context_chunks:
        # Jika tidak ada konteks yang relevan, berikan pesan yang jelas
        return """Maaf, saya tidak dapat menemukan informasi yang relevan dalam materi pembelajaran untuk menjawab pertanyaan Anda.

Silakan coba:
1. 🔄 Reformulasi pertanyaan dengan kata kunci yang berbeda
2. 📖 Periksa apakah topik tersebut ada dalam daftar pertemuan
3. 💭 Ajukan pertanyaan yang lebih spesifik tentang materi yang dipelajari

Saya hanya dapat menjawab berdasarkan materi pembelajaran yang tersedia dalam dataset."""
    
    # Jika ada konteks, gunakan konteks tersebut untuk generate jawaban
    context_string = "\n\n---\n\n".join(context_chunks)
    
    prompt = f"""Anda adalah asisten pembelajaran yang HANYA boleh menjawab berdasarkan KONTEKS MATERI yang diberikan.

ATURAN PENTING:
- WAJIB menggunakan HANYA informasi dari konteks materi di bawah
- DILARANG menggunakan pengetahuan umum atau informasi di luar konteks
- Jika informasi tidak lengkap dalam konteks, katakan "Berdasarkan materi yang tersedia..."
- Berikan jawaban yang terstruktur dan mudah dipahami
- Sertakan contoh dari konteks jika tersedia

KONTEKS MATERI:
---
{context_string}
---

PERTANYAAN MAHASISWA: "{query}"

INSTRUKSI JAWABAN:
Berikan penjelasan yang komprehensif berdasarkan HANYA konteks materi di atas. Jangan tambahkan informasi dari luar konteks.

JAWABAN ANDA:"""

    try:
        # Konfigurasi untuk respons yang lebih konsisten dan terfokus pada konteks
        generation_config = genai.types.GenerationConfig(
            temperature=0.2,  # Rendah untuk konsistensi dan mengurangi kreativitas
            top_p=0.8,        # Batasi variasi token
            top_k=40,         # Batasi pilihan token
            max_output_tokens=1500,  # Batas maksimal untuk jawaban yang fokus
        )
        
        response = llm_model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        if response.text:
            # Tambahkan disclaimer jika perlu
            answer = response.text
            
            # Cek apakah jawaban mengindikasikan penggunaan pengetahuan luar
            problematic_phrases = [
                "secara umum", "biasanya", "umumnya", "pada umumnya",
                "menurut teori", "dalam praktek umum", "secara teoritis"
            ]
            
            contains_external_knowledge = any(phrase in answer.lower() for phrase in problematic_phrases)
            
            if contains_external_knowledge:
                answer += "\n\n*️⃣ Catatan: Jawaban di atas berdasarkan materi pembelajaran yang tersedia dalam dataset.*"
            
            return answer
        else:
            return "Maaf, tidak dapat menghasilkan jawaban berdasarkan materi yang tersedia."
            
    except Exception as e:
        return f"Maaf, terjadi kesalahan teknis saat memproses materi: {e}"


def generate_mcq(pertemuan_id, resources, num_questions=3):
    """Generate MCQ berdasarkan HANYA materi dalam dataset untuk pertemuan tertentu."""
    
    # Ambil chunks yang relevan untuk pertemuan ini
    relevant_chunks = [
        c["chunk_text"] for c in resources["chunks_data"] 
        if str(c.get("pertemuan_id")) == str(pertemuan_id)
    ]
    
    if not relevant_chunks: 
        return []
    
    # Gabungkan konteks dengan batasan panjang
    context = "\n\n".join(relevant_chunks[:5])  # Ambil maksimal 5 chunks
    context = context[:10000]  # Batasi panjang konteks untuk efisiensi
    
    prompt = f"""Berdasarkan HANYA materi pembelajaran berikut, buatkan {num_questions} soal pilihan ganda (MCQ).

MATERI PEMBELAJARAN:
---
{context}
---

ATURAN PEMBUATAN SOAL:
- WAJIB berdasarkan HANYA konten materi di atas
- DILARANG menambahkan informasi di luar materi
- Fokus pada konsep kunci yang ada dalam materi
- Pastikan semua opsi jawaban masuk akal berdasarkan konteks

Format output dalam JSON list yang valid:
[
  {{
    "pertanyaan": "teks pertanyaan berdasarkan materi",
    "opsi": {{
      "A": "opsi A berdasarkan materi",
      "B": "opsi B berdasarkan materi", 
      "C": "opsi C berdasarkan materi",
      "D": "opsi D berdasarkan materi"
    }},
    "jawaban_benar": "A/B/C/D",
    "pembahasan": "penjelasan berdasarkan materi yang diberikan",
    "topik_terkait": "topik utama dari materi"
  }}
]

HANYA kembalikan JSON list, tidak ada teks lain."""

    try:
        # Konfigurasi untuk output yang lebih terprediksi
        generation_config = genai.types.GenerationConfig(
            temperature=0.3,  # Sedikit lebih tinggi untuk variasi soal
            top_p=0.9,
            top_k=50,
            max_output_tokens=2000,
        )
        
        response = resources["llm_model"].generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Extract JSON dari response
        json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
        if json_match:
            questions = json.loads(json_match.group(0))
            
            # Validasi bahwa soal berdasarkan materi
            valid_questions = []
            for q in questions:
                if all(key in q for key in ["pertanyaan", "opsi", "jawaban_benar", "pembahasan"]):
                    # Tambahan validasi: cek apakah pembahasan merujuk pada materi
                    if "berdasarkan materi" in q.get("pembahasan", "").lower() or \
                       any(keyword in context.lower() for keyword in q["pertanyaan"].lower().split()[:3]):
                        valid_questions.append(q)
            
            return valid_questions[:num_questions]  # Batasi sesuai permintaan
        return []
        
    except Exception as e:
        st.error(f"Gagal membuat soal berdasarkan materi: {e}")
        return []

# --- 3. Logika UI & State Management ---

def initialize_session_state():
    """Menginisialisasi semua session state yang dibutuhkan."""
    states = {
        "current_view": "course_selection", "selected_course_id": None, "course_resources": None,
        "current_pertemuan_id": None, "current_pertemuan_judul": "Pilih Pertemuan",
        "chat_history": [], "quiz_mode": None, "quiz_questions": [],
        "user_answers": {}, "current_question_index": 0, "auto_send_prompt_topic": None,
    }
    for key, value in states.items():
        if key not in st.session_state: st.session_state[key] = value

def ui_main():
    """Fungsi utama untuk menjalankan seluruh logika UI."""
    initialize_session_state()

    if st.session_state.current_view == "course_selection":
        st.title("🎓 Chatbot Mata Kuliah"); st.subheader("Pilih Mata Kuliah untuk Memulai")
        if not AVAILABLE_COURSES:
            st.error("Tidak ada mata kuliah ditemukan."); return
        for name, config in AVAILABLE_COURSES.items():
            if st.button(f"{name}", key=f"course_{config['id']}", use_container_width=True):
                st.session_state.current_view = "meeting_view"; st.session_state.selected_course_id = config["id"]
                st.rerun()

    elif st.session_state.current_view == "meeting_view":
        course_id = st.session_state.selected_course_id
        if st.session_state.course_resources is None or st.session_state.course_resources["config"]["id"] != course_id:
            with st.spinner("Memuat resource mata kuliah..."):
                st.session_state.course_resources = load_course_resources(course_id)
        
        resources = st.session_state.course_resources
        if not resources:
            st.error("Gagal memuat resource."); st.button("Kembali", on_click=lambda: st.session_state.update(current_view="course_selection")); return
        
        # Sidebar
        config = resources["config"]
        # Update sidebar section in ui_main()
        with st.sidebar:
            st.header(f"📖 {config['id'].replace('_', ' ').title()}")
            st.caption(f"Model AI: {config['model_type']}")
            
            # Enhanced system status
            st.markdown("### 🔧 System Status")
            
            # Model status
            if config.get('has_finetuned'):
                st.success("✅ Fine-tuned Embedding")
            else:
                st.info("📝 Base Embedding Model")
            
            # Reranker status
            if RERANKER_AVAILABLE:
                reranker = get_reranker()
                if reranker:
                    st.success("✅ Enhanced Reranker")
                else:
                    st.warning("⚠️ Reranker Fallback")
            else:
                st.error("❌ Reranker Unavailable")
            
            if st.button("⬅️ Ganti Mata Kuliah", use_container_width=True): 
                st.session_state.current_view = "course_selection"; st.rerun()
            
            st.divider()
            st.subheader("Navigasi Pertemuan")
            for p_data in resources["outline"]:
                p_id, p_title = p_data.get("id"), p_data.get("judul", "Tanpa Judul")
                btn_type = "primary" if str(st.session_state.get("current_pertemuan_id")) == str(p_id) else "secondary"
                if st.button(p_title, key=f"p_{p_id}", use_container_width=True, type=btn_type):
                    if str(st.session_state.get("current_pertemuan_id")) != str(p_id):
                        st.session_state.current_pertemuan_id, st.session_state.current_pertemuan_judul = p_id, p_title
                        st.session_state.chat_history, st.session_state.quiz_mode = [], None; st.rerun()
        
        # Konten Utama
        if not st.session_state.get("current_pertemuan_id"):
            st.info("👈 Silakan pilih pertemuan dari menu navigasi."); return

        st.title(f"📍 Pertemuan {st.session_state.current_pertemuan_id}: {st.session_state.current_pertemuan_judul}")
        
        # Tentukan tab default berdasarkan auto_send_prompt_topic
        default_tab = "Tanya Jawab AI" if st.session_state.auto_send_prompt_topic else "Materi"
        tabs_list = ["📜 Materi", "💬 Tanya Jawab AI", "📝 Kuis Pemahaman"]
        
        # Buat fungsi untuk mengubah tab
        def set_active_tab(tab_name):
            st.session_state.active_tab = tab_name

        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = default_tab
            
        tab_materi, tab_chat, tab_kuis = st.tabs(tabs_list)

        with tab_materi:
            materi_chunks = [c["chunk_text"] for c in resources["chunks_data"] if str(c.get("pertemuan_id")) == str(st.session_state.current_pertemuan_id)]
            st.markdown("\n\n---\n\n".join(materi_chunks) if materi_chunks else "Materi untuk pertemuan ini belum tersedia.")
        
        with tab_chat:
            ui_chat_rag(resources)
        
        with tab_kuis:
            ui_quiz(resources)

def ui_chat_rag(resources):
    """Enhanced chat interface dengan detailed ranking dan reranker functionality."""
    
    if not resources:
        st.error("❌ Resources tidak tersedia. Silakan muat ulang mata kuliah.")
        return
    
    # Enhanced info about reranking system
    # Enhanced info about models and reranking system
    with st.expander("🔧 Tentang Sistem AI & Model", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **📊 Sistem Ranking Multi-Level:**
            🎯 **Semantic Similarity** (50%) - Cross-encoder analysis
            🔤 **Keyword Relevance** (30%) - Keyword matching & density  
            📝 **Content Quality** (20%) - Length & structure analysis
            """)
        
        with col2:
            config = resources.get('config', {})
            if config.get('is_custom_pth'):
                st.success(f"""
                **🎯 Custom Model Active:**
                - Model: {config['model_type']}
                - File: {os.path.basename(config['embedding_model_path'])}
                - Status: Specialized untuk domain ini
                - Performance: Enhanced untuk topik spesifik
                """)
            else:
                st.info(f"""
                **📝 Model Info:**
                - Type: {config.get('model_type', 'Base Model')}
                - Base: {config.get('base_model_name', 'all-MiniLM-L6-v2')}
                - Status: {'Fine-tuned' if config.get('has_finetuned') else 'General Purpose'}
                """)
        
        # Model comparison results (if requested)
        if hasattr(st.session_state, 'show_model_comparison') and st.session_state.show_model_comparison:
            st.markdown("### 🔍 Model Performance Analysis")
            
            if config.get('is_custom_pth'):
                custom_loader = get_custom_model_loader()
                base_model = SentenceTransformer('all-MiniLM-L6-v2')
                
                comparison = custom_loader.compare_models(
                    base_model, 
                    resources['embedding_model'],
                    test_query="Jelaskan konsep sistem operasi"
                )
                
                if 'error' not in comparison:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Embedding Dimension", comparison['embedding_dim'])
                    with col2:
                        st.metric("Model Similarity", f"{comparison['similarity']:.3f}")
                    with col3:
                        similarity_status = "🔄 Different" if comparison['similarity'] < 0.95 else "➡️ Similar"
                        st.metric("Difference Status", similarity_status)
                    
                    with st.expander("📊 Detailed Embedding Comparison"):
                        st.json(comparison)
                else:
                    st.error(f"Comparison failed: {comparison['error']}")
            
            # Reset comparison flag
            st.session_state.show_model_comparison = False
        
        # Reranker status
        if RERANKER_AVAILABLE:
            reranker = get_reranker()
            if reranker:
                st.success("✅ **Enhanced Reranker**: Aktif - Menggunakan Cross-Encoder untuk ranking yang lebih akurat")
            else:
                st.warning("⚠️ **Reranker**: Fallback ke FAISS similarity")
        else:
            st.warning("⚠️ **Reranker**: Modul tidak tersedia, menggunakan FAISS similarity")
    
    # Handle auto-send prompt
    if st.session_state.auto_send_prompt_topic:
        auto_query = f"Tolong jelaskan lebih detail mengenai topik '{st.session_state.auto_send_prompt_topic}'."
        st.session_state.chat_history.append({"role": "user", "content": auto_query})
        
        try:
            # Use enhanced search with reranking
            search_results = search_relevant_chunks_with_reranking(
                auto_query, resources, top_k=5, similarity_threshold=0.1, use_reranking=True
            )
            context_chunks = [result['text'] for result in search_results]
            answer = get_rag_answer(auto_query, context_chunks, resources["llm_model"])
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        except Exception as e:
            error_msg = f"Maaf, terjadi kesalahan: {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        
        st.session_state.auto_send_prompt_topic = None
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): 
            st.markdown(msg["content"])
    
    # Handle new chat input
    if prompt := st.chat_input("Tanyakan sesuatu tentang materi pembelajaran..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"): 
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Mencari & meranking sumber terbaik dari materi..."):
                try:
                    # Enhanced search dengan detailed results
                    search_results = search_relevant_chunks_with_reranking(
                        prompt, resources, top_k=5, similarity_threshold=0.1, use_reranking=True
                    )
                    
                    if search_results:
                        context_chunks = [result['text'] for result in search_results]
                        answer = get_rag_answer(prompt, context_chunks, resources["llm_model"])
                        st.markdown(answer)
                        
                        # Enhanced source display dengan detailed ranking
                        st.markdown("---")
                        st.markdown("### 📊 **Analisis Sumber & Ranking**")
                        
                        # Summary metrics
                        avg_score = np.mean([r['scores']['final_score'] for r in search_results])
                        top_grade = search_results[0]['relevance_grade'] if search_results else "N/A"
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📈 Avg Score", f"{avg_score:.2f}")
                        with col2:
                            st.metric("🏆 Top Grade", top_grade.split()[1] if ' ' in top_grade else top_grade)
                        with col3:
                            st.metric("📚 Sources Used", len(search_results))
                        
                        # Detailed ranking table
                        with st.expander("🔍 **Detailed Source Ranking & Analysis**", expanded=False):
                            for i, result in enumerate(search_results):
                                scores = result['scores']
                                metadata = result['metadata']
                                
                                st.markdown(f"#### **#{result['rank']} {result['relevance_grade']}** (Score: {scores['final_score']:.3f})")
                                
                                # Score breakdown dengan progress bars
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.markdown("**📊 Score Breakdown:**")
                                    st.progress(scores['semantic_similarity'], text=f"🧠 Semantic: {scores['semantic_similarity']:.3f}")
                                    st.progress(scores['keyword_relevance'], text=f"🔤 Keywords: {scores['keyword_relevance']:.3f}")
                                    st.progress(scores['content_quality'], text=f"📝 Quality: {scores['content_quality']:.3f}")
                                
                                with col2:
                                    st.markdown("**📋 Metadata:**")
                                    if 'pertemuan_id' in metadata:
                                        st.caption(f"📖 Pertemuan: {metadata['pertemuan_id']}")
                                    if 'chunk_id' in metadata:
                                        st.caption(f"🔖 Chunk: {metadata['chunk_id']}")
                                    if 'faiss_score' in metadata:
                                        st.caption(f"⚡ FAISS: {metadata['faiss_score']:.3f}")
                                
                                # Content preview
                                content_preview = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
                                st.text_area(f"Content Preview #{result['rank']}", content_preview, height=80, disabled=True, key=f"preview_{i}")
                                
                                if i < len(search_results) - 1:
                                    st.divider()
                        
                        # Reranker comparison (if available)
                        if RERANKER_AVAILABLE and hasattr(st.session_state, 'last_search_results'):
                            with st.expander("🔄 **Reranker Impact Analysis**", expanded=False):
                                st.info("""
                                **Reranker Benefits:**
                                - 🎯 **Semantic Understanding**: Better captures query intent
                                - 📊 **Multi-factor Scoring**: Combines similarity, keywords & quality
                                - 🔄 **Re-ordering**: Improves ranking over FAISS-only results
                                - 📈 **Relevance Grades**: Clear quality indicators
                                """)
                                
                                # Show reranking effect
                                for i, result in enumerate(search_results[:3]):
                                    metadata = result['metadata']
                                    faiss_rank = metadata.get('faiss_rank', 'N/A')
                                    current_rank = result['rank']
                                    
                                    rank_change = ""
                                    if faiss_rank != 'N/A':
                                        change = faiss_rank - current_rank
                                        if change > 0:
                                            rank_change = f"⬆️ +{change}"
                                        elif change < 0:
                                            rank_change = f"⬇️ {change}"
                                        else:
                                            rank_change = "➡️ Same"
                                    
                                    st.caption(f"**Source #{current_rank}**: FAISS #{faiss_rank} → Reranked #{current_rank} {rank_change}")
                    
                    else:
                        st.warning("⚠️ Tidak ditemukan sumber yang relevan dalam dataset")
                        st.info("💡 **Saran untuk hasil yang lebih baik:**\n"
                                "- Gunakan kata kunci spesifik dari materi\n"
                                "- Coba sinonim atau istilah alternatif\n"
                                "- Periksa ejaan dan format pertanyaan")
                    
                    if search_results:
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"Maaf, terjadi kesalahan saat mencari dalam materi: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

def ui_quiz(resources):
    """Menampilkan antarmuka kuis dan logikanya."""
    if st.session_state.quiz_mode is None:
        if st.button("💡 Mulai Kuis", type="primary", use_container_width=True): st.session_state.quiz_mode = "generating"; st.rerun()
    
    elif st.session_state.quiz_mode == "generating":
        with st.spinner("Membuat soal..."):
            st.session_state.quiz_questions = generate_mcq(st.session_state.current_pertemuan_id, resources)
            if st.session_state.quiz_questions:
                st.session_state.quiz_mode = "ongoing"; st.session_state.user_answers = {}; st.session_state.current_question_index = 0
            else:
                st.error("Gagal membuat soal."); st.session_state.quiz_mode = None
            st.rerun()
            
    elif st.session_state.quiz_mode == "ongoing":
        q_idx = st.session_state.current_question_index
        if q_idx >= len(st.session_state.quiz_questions): st.session_state.quiz_mode = "results"; st.rerun()
        q_data = st.session_state.quiz_questions[q_idx]
        st.markdown(f"**Soal {q_idx + 1}/{len(st.session_state.quiz_questions)}:** {q_data['pertanyaan']}")
        user_choice = st.radio("Pilih:", [f"{k}. {v}" for k, v in q_data.get("opsi", {}).items()], key=f"q_{q_idx}")
        if st.button("Kunci & Lanjut", key=f"submit_{q_idx}"):
            st.session_state.user_answers[q_idx] = user_choice.split(".")[0]
            st.session_state.current_question_index += 1; st.rerun()

    elif st.session_state.quiz_mode == "results":
        st.subheader("📊 Hasil Kuis")
        score = sum(1 for i, q in enumerate(st.session_state.quiz_questions) if st.session_state.user_answers.get(i) == q.get("jawaban_benar"))
        st.metric("Skor Anda", f"{score}/{len(st.session_state.quiz_questions)}")
        with st.expander("Lihat Pembahasan", expanded=True):
            for idx, q in enumerate(st.session_state.quiz_questions):
                st.markdown(f"**Soal {idx+1}:** {q['pertanyaan']}")
                is_correct = st.session_state.user_answers.get(idx) == q.get("jawaban_benar")
                st.markdown(f"Jawaban Anda: `{st.session_state.user_answers.get(idx)}` {'✅' if is_correct else '❌'}")
                st.markdown(f"Jawaban Benar: `{q['jawaban_benar']}`")
                st.info(f"Pembahasan: {q.get('pembahasan', 'Tidak ada.')}")
                if not is_correct:
                    topic = q.get("topik_terkait", "topik ini")
                    if st.button(f"💬 Tanya AI tentang '{topic}'", key=f"ask_{idx}"):
                        st.session_state.auto_send_prompt_topic = topic
                        st.session_state.quiz_mode = None # Keluar dari mode kuis
                        st.rerun() # Ini akan memicu logika auto-send di tab chat
                st.divider()
        if st.button("Ulangi Kuis"): st.session_state.quiz_mode = None; st.rerun()


if __name__ == "__main__":
    ui_main()