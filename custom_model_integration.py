import torch
import os
from sentence_transformers import SentenceTransformer
import streamlit as st
import numpy as np
from typing import Optional, Dict, Any

class CustomModelLoader:
    """Loader untuk mengintegrasikan custom fine-tuned .pth models"""
    
    def __init__(self, base_model_name='all-MiniLM-L6-v2'):
        self.base_model_name = base_model_name
        self.device = torch.device("cpu")  # Use CPU for compatibility
    
    def load_custom_pth_model(self, pth_file_path: str) -> Optional[SentenceTransformer]:
        """Load custom fine-tuned model dari .pth file"""
        try:
            if not os.path.exists(pth_file_path):
                st.warning(f"Custom model file {pth_file_path} tidak ditemukan")
                return None
            
            # Load base model architecture
            st.info(f"🔄 Loading base model architecture: {self.base_model_name}")
            base_model = SentenceTransformer(self.base_model_name, device=self.device)
            
            # Load custom weights
            st.info(f"🔄 Loading custom weights dari: {pth_file_path}")
            checkpoint = torch.load(pth_file_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                # Direct state dict
                state_dict = checkpoint
            
            # Try to load state dict ke model
            try:
                # Method 1: Direct loading
                base_model.load_state_dict(state_dict, strict=False)
                st.success(f"✅ Custom model berhasil dimuat dengan direct loading")
                return base_model
                
            except Exception as direct_error:
                st.warning(f"Direct loading gagal: {direct_error}")
                
                # Method 2: Load to transformer component
                try:
                    # Get transformer component (usually the first module)
                    transformer_component = None
                    for name, module in base_model.named_modules():
                        if hasattr(module, 'auto_model'):  # SentenceTransformer's transformer
                            transformer_component = module
                            break
                    
                    if transformer_component:
                        # Filter state dict untuk transformer
                        filtered_state_dict = {}
                        for key, value in state_dict.items():
                            # Remove module prefixes that might not match
                            clean_key = key
                            for prefix in ['module.', 'model.', '0.']:
                                if clean_key.startswith(prefix):
                                    clean_key = clean_key[len(prefix):]
                            filtered_state_dict[clean_key] = value
                        
                        transformer_component.auto_model.load_state_dict(filtered_state_dict, strict=False)
                        st.success(f"✅ Custom model berhasil dimuat ke transformer component")
                        return base_model
                    else:
                        st.error("Transformer component tidak ditemukan")
                        return None
                        
                except Exception as component_error:
                    st.error(f"Component loading juga gagal: {component_error}")
                    return None
                    
        except Exception as e:
            st.error(f"Error loading custom model: {e}")
            return None
    
    def validate_custom_model(self, model: SentenceTransformer, test_texts: list = None) -> bool:
        """Validate bahwa custom model berfungsi dengan baik"""
        if test_texts is None:
            test_texts = ["Test embedding generation", "Sistem operasi adalah software"]
        
        try:
            embeddings = model.encode(test_texts)
            if embeddings is not None and len(embeddings) > 0:
                embedding_dim = embeddings.shape[1]
                st.success(f"✅ Model validation successful:")
                st.info(f"   - Embedding dimension: {embedding_dim}")
                st.info(f"   - Test embeddings shape: {embeddings.shape}")
                return True
            return False
        except Exception as e:
            st.error(f"Model validation failed: {e}")
            return False
    
    def compare_models(self, base_model: SentenceTransformer, custom_model: SentenceTransformer, 
                      test_query: str = "Apa itu sistem operasi?") -> Dict[str, Any]:
        """Compare base model vs custom model embeddings"""
        try:
            base_embedding = base_model.encode([test_query])
            custom_embedding = custom_model.encode([test_query])
            
            # Calculate similarity
            similarity = np.dot(base_embedding[0], custom_embedding[0]) / (
                np.linalg.norm(base_embedding[0]) * np.linalg.norm(custom_embedding[0])
            )
            
            return {
                'base_embedding_sample': base_embedding[0][:5].tolist(),  # First 5 dims
                'custom_embedding_sample': custom_embedding[0][:5].tolist(),
                'similarity': float(similarity),
                'embedding_dim': len(base_embedding[0])
            }
        except Exception as e:
            return {'error': str(e)}

# Global loader instance
@st.cache_resource
def get_custom_model_loader():
    """Get cached custom model loader instance"""
    return CustomModelLoader()