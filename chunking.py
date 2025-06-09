import os
import re
import glob
import json

def clean_filename(title):
    """Membersihkan judul agar menjadi nama file yang valid."""
    title = title.strip().lstrip('#').strip()
    title = re.sub(r'[\s/\\:]+', '_', title)
    title = re.sub(r'[^\w\-]', '', title)
    return title

def discover_and_parse_outlines():
    """
    Secara dinamis menemukan dan mem-parsing semua file 'outline_*.txt'
    di direktori saat ini untuk membangun struktur data mata kuliah.
    """
    print("Mencari dan mem-parsing file outline...")
    discovered_matakuliah = []
    
    # Fix 1: Search in the correct pattern and locations
    search_patterns = [
        'outline_*.txt',  # Current directory
        'dataset/*/outline_*.txt',  # Dataset subdirectories
        'dataset/*/*/outline_*.txt'  # Nested subdirectories
    ]
    
    outline_files = []
    for pattern in search_patterns:
        outline_files.extend(glob.glob(pattern))
    
    if not outline_files:
        print("  Tidak ditemukan file outline. Mencoba pola yang lebih spesifik...")
        # Try specific known location
        specific_files = [
            'dataset/SistemOperasi/outline_operating_systems.txt',
            'dataset/ArtificialIntelligence/outline_ai.txt',
            'dataset/SistemInformasi/outline_sistem_informasi.txt',
            'dataset/StrukturData/outline_struktur_data.txt'
        ]
        for file_path in specific_files:
            if os.path.exists(file_path):
                outline_files.append(file_path)
    
    for outline_file in outline_files:
        print(f"  -> Ditemukan: {outline_file}")
        try:
            with open(outline_file, 'r', encoding='utf-8') as f:
                content = f.read()

                # Ekstrak nama mata kuliah
                matkul_name_match = re.search(r'MATAKULIAH:\s*(.*)', content)
                if not matkul_name_match:
                    print(f"    Peringatan: Tidak dapat menemukan 'MATAKULIAH:' di {outline_file}. Melewati file ini.")
                    continue
                
                matkul_name = clean_filename(matkul_name_match.group(1))
                
                # Get directory path of outline file
                outline_dir = os.path.dirname(outline_file)
                
                # Parse pertemuan with more detailed info
                pertemuan_blocks = re.split(r'\nPERTEMUAN:', content)
                pertemuan_list = []
                
                for block in pertemuan_blocks[1:]:  # Skip first block (header)
                    if not block.strip():
                        continue
                    
                    current_pertemuan = {}
                    lines = block.strip().splitlines()
                    
                    if lines:
                        # Extract pertemuan ID from first line
                        pertemuan_id_match = re.match(r'^\s*(\d+)', lines[0])
                        if pertemuan_id_match:
                            current_pertemuan['id'] = int(pertemuan_id_match.group(1))
                        
                        # Parse other fields
                        for line in lines:
                            if ":" in line:
                                key, value = line.split(":", 1)
                                key_clean = key.strip().lower().replace(" ", "_")
                                value_clean = value.strip()
                                current_pertemuan[key_clean] = value_clean
                    
                    if 'file_materi' in current_pertemuan:
                        # Split path and file
                        file_path = current_pertemuan['file_materi']
                        dir_path = os.path.dirname(file_path)
                        file_name = os.path.basename(file_path)
                        
                        current_pertemuan['path'] = dir_path
                        current_pertemuan['file'] = file_name
                        pertemuan_list.append(current_pertemuan)
                
                discovered_matakuliah.append({
                    "name": matkul_name,
                    "outline_dir": outline_dir,
                    "pertemuan": pertemuan_list
                })
                print(f"    -> Berhasil mem-parsing {matkul_name} dengan {len(pertemuan_list)} pertemuan.")

        except Exception as e:
            print(f"    Error saat membaca {outline_file}: {e}")
            
    return discovered_matakuliah

def extract_content_to_chunks(content, pertemuan_data):
    """
    Extract content and convert to chunks with metadata like the example JSON
    """
    chunks = []
    
    # Split by headings (### or ##)
    sections = re.split(r'(###?\s+.*?)(?=\n)', content, flags=re.MULTILINE)
    
    current_heading = "Umum"
    accumulated_text = ""
    chunk_counter = 0
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Check if this is a heading
        heading_match = re.match(r'^(###?\s+.*)', section)
        if heading_match:
            # Save previous chunk if exists
            if accumulated_text.strip():
                chunk = create_chunk_metadata(
                    accumulated_text.strip(), 
                    current_heading, 
                    pertemuan_data, 
                    chunk_counter
                )
                chunks.append(chunk)
                chunk_counter += 1
            
            current_heading = heading_match.group(1)
            accumulated_text = ""
        else:
            accumulated_text += section + "\n"
    
    # Don't forget the last chunk
    if accumulated_text.strip():
        chunk = create_chunk_metadata(
            accumulated_text.strip(), 
            current_heading, 
            pertemuan_data, 
            chunk_counter
        )
        chunks.append(chunk)
    
    return chunks

def create_chunk_metadata(text, heading, pertemuan_data, chunk_counter):
    """
    Create chunk metadata similar to the example JSON structure
    """
    # Split large chunks if needed (max 1000 chars)
    max_chunk_size = 1000
    if len(text) <= max_chunk_size:
        return {
            "pertemuan_id": pertemuan_data.get('id', 0),
            "pertemuan_judul": pertemuan_data.get('judul', 'Unknown'),
            "original_heading": heading,
            "chunk_text": text,
            "chunk_id": f"p{pertemuan_data.get('id', 0)}_s{chunk_counter}_sc0"
        }
    else:
        # For now, just truncate. You can implement sliding window here
        return {
            "pertemuan_id": pertemuan_data.get('id', 0),
            "pertemuan_judul": pertemuan_data.get('judul', 'Unknown'),
            "original_heading": heading,
            "chunk_text": text[:max_chunk_size],
            "chunk_id": f"p{pertemuan_data.get('id', 0)}_s{chunk_counter}_sc0"
        }

def chunk_materi_to_json(file_path, base_dir, pertemuan_data):
    """
    Read material file and convert to JSON chunks with metadata
    """
    chunks = []
    
    # Resolve the full path
    if base_dir:
        full_path = os.path.join(base_dir, file_path)
    else:
        full_path = file_path
    
    if not os.path.exists(full_path):
        print(f"    Peringatan: File materi tidak ditemukan di '{full_path}'.")
        # Try alternative path constructions
        alternatives = [
            os.path.join("dataset", "SistemOperasi", file_path),
            os.path.join("dataset", "ArtificialIntelligence", file_path),
            os.path.join("dataset", "SistemInformasi", file_path),
            os.path.join("dataset", "StrukturData", file_path)
        ]
        
        for alt_path in alternatives:
            if os.path.exists(alt_path):
                full_path = alt_path
                print(f"    -> Ditemukan di: {full_path}")
                break
        else:
            return chunks

    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
            chunks = extract_content_to_chunks(content, pertemuan_data)
    except Exception as e:
        print(f"    Error saat memproses chunk dari {full_path}: {e}")

    return chunks

def create_dataset():
    """Fungsi utama untuk membuat seluruh dataset dalam format JSON."""
    # Langkah 1: Secara dinamis temukan dan bangun daftar mata kuliah dari file outline
    semua_matakuliah = discover_and_parse_outlines()
    
    if not semua_matakuliah:
        print("\nTidak ada outline mata kuliah yang valid ditemukan. Proses dihentikan.")
        print("Pastikan ada file 'outline_*.txt' dan skrip bash sudah dijalankan untuk membuat materi.")
        return

    base_dataset_dir = "dataset"
    if not os.path.exists(base_dataset_dir):
        os.makedirs(base_dataset_dir)
    print(f"\nFolder dataset utama akan disimpan di: '{base_dataset_dir}'")

    print("\nMemulai proses chunking untuk setiap mata kuliah yang ditemukan...")
    for matkul in semua_matakuliah:
        matkul_name = matkul["name"]
        outline_dir = matkul["outline_dir"]
        
        print(f"\n--- Memproses Mata Kuliah: {matkul_name} ---")
        
        # Collect all chunks for this mata kuliah
        all_chunks = []
        
        for pertemuan in matkul["pertemuan"]:
            # Gabungkan path direktori dan file dari hasil parsing
            file_path = os.path.join(pertemuan["path"], pertemuan["file"])
            
            print(f"  Membaca dan memecah file: {file_path}")
            chunks = chunk_materi_to_json(file_path, outline_dir, pertemuan)
            
            if not chunks:
                print(f"  Tidak ada chunk yang dihasilkan dari {file_path}.")
                continue
            
            all_chunks.extend(chunks)
            print(f"    -> Berhasil membuat {len(chunks)} chunks")

        # Save all chunks to JSON file in the mata kuliah directory
        if all_chunks:
            json_filename = f"processed_chunks_metadata_{matkul_name.lower()}.json"
            json_filepath = os.path.join(outline_dir, json_filename)
            
            try:
                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(all_chunks, f, ensure_ascii=False, indent=2)
                print(f"  -> JSON chunks disimpan ke: {json_filepath}")
                print(f"  -> Total chunks untuk {matkul_name}: {len(all_chunks)}")
                
                # Also create base and finetuned versions for compatibility
                base_filename = f"processed_chunks_metadata_base.json"
                base_filepath = os.path.join(outline_dir, base_filename)
                
                with open(base_filepath, 'w', encoding='utf-8') as f:
                    json.dump(all_chunks, f, ensure_ascii=False, indent=2)
                
                finetuned_filename = f"processed_chunks_metadata_finetuned.json" 
                finetuned_filepath = os.path.join(outline_dir, finetuned_filename)
                
                with open(finetuned_filepath, 'w', encoding='utf-8') as f:
                    json.dump(all_chunks, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                print(f"    -> Gagal menyimpan JSON {json_filepath}: {e}")
        else:
            print(f"  -> Tidak ada chunks yang dihasilkan untuk {matkul_name}")

    print("\nProses chunking dataset selesai.")
    print("\nFile JSON yang dihasilkan:")
    for matkul in semua_matakuliah:
        json_path = os.path.join(matkul["outline_dir"], f"processed_chunks_metadata_{matkul['name'].lower()}.json")
        if os.path.exists(json_path):
            print(f"  - {json_path}")

# --- JALANKAN PROSES ---
if __name__ == "__main__":
    # Penting: Pastikan Anda telah menjalankan skrip bash dari respons sebelumnya
    # untuk membuat semua folder dan file materi terlebih dahulu.
    create_dataset()
