import streamlit as st
import json
import faiss
import numpy as np
import os
import re
import glob
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
import torch

# --- 0. Dynamic Course Discovery ---

def discover_available_courses():
    """Dynamically discover all available courses from dataset folder."""
    courses = {}
    dataset_base = "dataset"
    
    # Check for global fine-tuned model (outside dataset folder)
    global_finetuned_paths = [
        "finetuned_embedding_model",
        "fine_tuned_model", 
        "custom_model",
        "trained_model"
    ]
    
    global_finetuned_model = None
    for path in global_finetuned_paths:
        if os.path.exists(path) and os.path.isdir(path):
            # Verify it's a valid SentenceTransformer model
            if os.path.exists(os.path.join(path, "config.json")):
                global_finetuned_model = path
                break
    
    if not os.path.exists(dataset_base):
        st.error(f"Dataset folder '{dataset_base}' not found!")
        return courses
    
    # Scan each subdirectory in dataset
    for course_dir in os.listdir(dataset_base):
        course_path = os.path.join(dataset_base, course_dir)
        
        if not os.path.isdir(course_path):
            continue
            
        # Look for required files
        json_files = glob.glob(os.path.join(course_path, "processed_chunks_metadata_*.json"))
        outline_files = glob.glob(os.path.join(course_path, "outline_*.txt"))
        
        if not json_files:
            continue
            
        # Determine course display name
        course_display_name = course_dir.replace("_", " ").title()
        if outline_files:
            try:
                with open(outline_files[0], 'r', encoding='utf-8') as f:
                    content = f.read()
                    matkul_match = re.search(r'MATAKULIAH:\s*(.*)', content)
                    if matkul_match:
                        course_display_name = matkul_match.group(1).strip()
            except:
                pass
        
        # Check for course-specific fine-tuned model first, then global
        finetuned_model_path = None
        course_specific_paths = [
            os.path.join(course_path, f"finetuned_embedding_model_{course_dir.lower()}"),
            os.path.join(course_path, "finetuned_embedding_model"),
            os.path.join(course_path, "fine_tuned_model")
        ]
        
        # First try course-specific models
        for path in course_specific_paths:
            if os.path.exists(path) and os.path.isdir(path):
                if os.path.exists(os.path.join(path, "config.json")):
                    finetuned_model_path = path
                    break
        
        # If no course-specific model, use global model
        if not finetuned_model_path and global_finetuned_model:
            finetuned_model_path = global_finetuned_model
        
        # Prefer fine-tuned files if model is available
        preferred_json = json_files[0]  # Default
        preferred_faiss = os.path.join(course_path, "vector_store.index")  # Default
        
        if finetuned_model_path:
            # Look for fine-tuned specific files
            finetuned_json = os.path.join(course_path, "processed_chunks_metadata_finetuned.json")
            if os.path.exists(finetuned_json):
                preferred_json = finetuned_json
            
            finetuned_faiss = os.path.join(course_path, "vector_store_finetuned.index")
            if os.path.exists(finetuned_faiss):
                preferred_faiss = finetuned_faiss
        
        # Configure course
        model_type = "Global Fine-tuned" if finetuned_model_path == global_finetuned_model else "Course-specific Fine-tuned" if finetuned_model_path else "Base"
        
        courses[course_display_name] = {
            "id": course_dir.lower(),
            "folder": course_dir,
            "chunks_json": preferred_json,
            "embedding_model_path": finetuned_model_path or 'all-MiniLM-L6-v2',
            "base_model_name": 'all-MiniLM-L6-v2',
            "faiss_index": preferred_faiss,
            "has_finetuned": finetuned_model_path is not None,
            "model_type": model_type,
            "is_global_model": finetuned_model_path == global_finetuned_model
        }
    
    return courses

# --- Configuration ---

st.set_page_config(page_title="Platform Edukasi AI", layout="wide", initial_sidebar_state="expanded")
load_dotenv() 

# API Key setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Variabel lingkungan GEMINI_API_KEY tidak ditemukan. Harap atur di Streamlit Secrets atau file .env.")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Gagal mengkonfigurasi Gemini API: {e}")
    st.stop()

# Dynamic course discovery
AVAILABLE_COURSES = discover_available_courses()
LLM_MODEL_NAME = "gemini-1.5-flash-latest"
device = torch.device("cpu")

# --- 1. Resource Loading Functions ---

@st.cache_resource
def load_embedding_model(course_config):
    """Load embedding model (fine-tuned or base)."""
    model_path = course_config["embedding_model_path"]
    base_model_name = course_config["base_model_name"]
    model_type = course_config.get("model_type", "Base")
    
    st.write(f"Memuat model embedding ({model_type})...")
    
    # Try to load fine-tuned model first
    if course_config["has_finetuned"] and os.path.exists(model_path):
        try:
            # Verify it's a valid SentenceTransformer directory
            if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
                model = SentenceTransformer(model_path)
                if course_config.get('is_global_model'):
                    st.success(f"✅ Global fine-tuned model berhasil dimuat dari `{os.path.basename(model_path)}`")
                else:
                    st.success(f"✅ Course-specific fine-tuned model berhasil dimuat dari `{os.path.basename(model_path)}`")
                return model, "fine-tuned"
            else:
                st.warning(f"⚠️ Path {model_path} bukan SentenceTransformer model yang valid")
        except Exception as e:
            st.warning(f"⚠️ Gagal memuat model fine-tuned: {e}. Menggunakan model dasar.")
    
    # Fallback to base model
    try:
        model = SentenceTransformer(base_model_name)
        st.info("ℹ️ Menggunakan model dasar.")
        return model, "base"
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None, "error"

@st.cache_resource
def load_course_resources(course_id):
    """Load all resources for a specific course."""
    course_name_display = next((c_name for c_name, c_info in AVAILABLE_COURSES.items() if c_info["id"] == course_id), None)
    if not course_name_display:
        return None
    
    course_config = AVAILABLE_COURSES[course_name_display]
    resources = {"course_name_display": course_name_display}
    
    # 1. Load embedding model
    model, model_type = load_embedding_model(course_config)
    if not model:
        return None
    resources["query_embedding_model"] = model
    resources["model_type"] = model_type

    # 2. Load chunks data
    try:
        with open(course_config["chunks_json"], "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
            resources["text_chunks_with_metadata"] = chunks_data
            st.info(f"📚 Berhasil memuat {len(chunks_data)} chunks materi")
    except Exception as e:
        st.error(f"❌ Gagal memuat file materi: {e}")
        return None

    # 3. Load or create FAISS Index
    faiss_path = course_config["faiss_index"]
    if os.path.exists(faiss_path):
        st.write("📊 Memuat FAISS Index yang sudah ada...")
        resources["faiss_index"] = faiss.read_index(faiss_path)
        st.success(f"✅ FAISS Index berhasil dimuat ({resources['faiss_index'].ntotal} vectors)")
    else:
        st.warning(f"⚠️ FAISS Index tidak ditemukan. Membuat index baru...")
        passages = [chunk['chunk_text'] for chunk in resources["text_chunks_with_metadata"]]
        if passages:
            with st.spinner("🔄 Membuat embedding untuk semua materi..."):
                corpus_embeddings = model.encode(passages, convert_to_tensor=True, device=device)
            
            embedding_dim = corpus_embeddings.shape[1]
            index = faiss.IndexFlatIP(embedding_dim)
            index.add(corpus_embeddings.cpu().numpy())
            resources["faiss_index"] = index
            
            os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
            faiss.write_index(index, faiss_path)
            st.success(f"✅ FAISS Index baru berhasil dibuat dan disimpan ({index.ntotal} vectors)")

    # 4. Load LLM model
    resources["llm_model"] = genai.GenerativeModel(model_name=LLM_MODEL_NAME)
    
    st.success(f"🎉 Semua resource untuk '{course_name_display}' siap digunakan!")
    return resources

# --- 2. Core Functions ---

def search_relevant_chunks(user_query_text, app_resources, top_k=5):
    """Search for relevant chunks using the loaded model."""
    model = app_resources["query_embedding_model"]
    faiss_idx = app_resources["faiss_index"]
    text_chunks_meta = app_resources["text_chunks_with_metadata"]

    if not all([model, faiss_idx, text_chunks_meta]):
        return []
    
    with torch.no_grad():
        query_embedding = model.encode(user_query_text, convert_to_tensor=True, device=device)
    
    query_np = np.expand_dims(query_embedding.cpu().numpy(), axis=0)
    
    if faiss_idx.ntotal == 0:
        return []

    distances, indices = faiss_idx.search(query_np, min(top_k, faiss_idx.ntotal))
    
    return [text_chunks_meta[i]["chunk_text"] for i in indices[0] if i != -1]

def get_rag_answer(user_query, context_chunks, llm_model):
    """Generate answer from LLM based on context."""
    if not llm_model:
        return "Error: Model LLM tidak siap."
    
    if not context_chunks:
        prompt = f"Jawab pertanyaan berikut berdasarkan pengetahuan umum Anda: \"{user_query}\""
    else:
        context_string = "\n\n---\n\n".join(context_chunks)
        prompt = f"""Berdasarkan KONTEKS MATERI di bawah ini, jawablah PERTANYAAN MAHASISWA dengan jelas dan akurat.

KONTEKS MATERI:
---
{context_string}
---

PERTANYAAN MAHASISWA: "{user_query}"

JAWABAN:"""
    
    try:
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Maaf, terjadi kesalahan: {e}"

# --- 3. Main UI ---

st.title("🤖 Platform Edukasi AI")
st.caption("Sistem pembelajaran cerdas dengan RAG yang mendukung multiple mata kuliah")

# Course selection in sidebar
with st.sidebar:
    st.header("📚 Pilih Mata Kuliah")
    
    if not AVAILABLE_COURSES:
        st.error("❌ Tidak ada mata kuliah yang ditemukan!")
        st.info("💡 Pastikan folder dataset berisi subdirektori dengan file JSON materi")
        st.stop()
    
    course_options = list(AVAILABLE_COURSES.keys())
    selected_course_name = st.selectbox(
        "Mata Kuliah:",
        course_options,
        index=0
    )
    
    # Display course info with enhanced model information
    course_info = AVAILABLE_COURSES[selected_course_name]
    model_icon = "🌐" if course_info.get('is_global_model') else "🔧" if course_info['has_finetuned'] else "📝"
    
    st.info(f"""
    **📁 Folder:** {course_info['folder']}
    **{model_icon} Model:** {course_info.get('model_type', 'Base')}
    **📊 Chunks:** {os.path.basename(course_info['chunks_json'])}
    **🎯 FAISS:** {os.path.basename(course_info['faiss_index'])}
    """)
    
    # Show global model info if applicable
    if course_info.get('is_global_model'):
        st.success(f"🌐 Menggunakan model global: `{course_info['embedding_model_path']}`")
    
    # Update session state when course changes
    selected_course_id = course_info["id"]
    if st.session_state.get("selected_course_id") != selected_course_id:
        st.session_state.selected_course_id = selected_course_id
        if "messages" in st.session_state:
            st.session_state.messages = []
        st.rerun()

# Initialize default course if not set
if "selected_course_id" not in st.session_state:
    st.session_state.selected_course_id = course_info["id"]

# Load resources for selected course
app_res = load_course_resources(st.session_state.selected_course_id)

if app_res and app_res.get("query_embedding_model"):
    st.header("💬 Chat dengan Asisten AI")
    st.info(f"🎯 **{app_res['course_name_display']}** | Model: **{app_res['model_type']}**")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Tanyakan sesuatu tentang materi..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Mencari jawaban..."):
                # 1. Search relevant chunks
                relevant_chunks = search_relevant_chunks(prompt, app_res)
                
                # 2. Generate answer from LLM
                response = get_rag_answer(prompt, relevant_chunks, app_res["llm_model"])
                
                # Display answer
                st.markdown(response)
                
                # Show sources used (optional)
                if relevant_chunks:
                    with st.expander("📖 Lihat sumber yang digunakan"):
                        for i, chunk in enumerate(relevant_chunks[:3]):
                            st.info(f"**Sumber {i+1}:**\n\n{chunk.strip()[:500]}...")

        # Save assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.error("❌ Gagal memuat resource aplikasi. Periksa file dan coba lagi.")
    
with st.sidebar:
    st.header("🔍 Courses Ditemukan")
    for course_name, course_info in AVAILABLE_COURSES.items():
        if course_info.get('is_global_model'):
            status_icon = "🌐"
        elif course_info['has_finetuned']:
            status_icon = "🔧"
        else:
            status_icon = "📝"
        st.text(f"{status_icon} {course_name}")
    
    # Show legend
    st.caption("""
    🌐 Global Fine-tuned Model
    🔧 Course-specific Fine-tuned
    📝 Base Model
    """)