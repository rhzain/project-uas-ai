import os
import numpy as np
import streamlit as st
from sentence_transformers import CrossEncoder
import torch
from typing import List, Tuple, Dict, Any

class EnhancedReranker:
    """Reranker untuk meningkatkan kualitas ranking hasil retrieval"""
    
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model_name = model_name
        self.model = None
        self.device = torch.device("cpu")  # Use CPU for compatibility
        
    def load_model(self):
        """Load cross-encoder model for reranking"""
        try:
            self.model = CrossEncoder(self.model_name, device=self.device)
            return True
        except Exception as e:
            st.warning(f"Failed to load reranker model: {e}")
            return False
    
    def calculate_semantic_similarity(self, query: str, chunks: List[str]) -> List[float]:
        """Calculate semantic similarity scores using cross-encoder"""
        if not self.model:
            if not self.load_model():
                # Fallback to simple similarity if model fails
                return [0.5] * len(chunks)
        
        try:
            # Create query-passage pairs
            pairs = [[query, chunk] for chunk in chunks]
            
            # Get cross-encoder scores
            scores = self.model.predict(pairs)
            
            # Normalize scores to 0-1 range
            scores = np.array(scores)
            min_score, max_score = scores.min(), scores.max()
            if max_score > min_score:
                normalized_scores = (scores - min_score) / (max_score - min_score)
            else:
                normalized_scores = scores
                
            return normalized_scores.tolist()
            
        except Exception as e:
            st.warning(f"Error in reranking: {e}")
            return [0.5] * len(chunks)
    
    def calculate_keyword_relevance(self, query: str, chunks: List[str]) -> List[float]:
        """Calculate keyword-based relevance scores"""
        query_words = set(query.lower().split())
        relevance_scores = []
        
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            
            # Jaccard similarity
            intersection = len(query_words.intersection(chunk_words))
            union = len(query_words.union(chunk_words))
            jaccard_score = intersection / union if union > 0 else 0
            
            # Keyword density
            keyword_matches = sum(1 for word in query_words if word in chunk.lower())
            keyword_density = keyword_matches / len(query_words) if len(query_words) > 0 else 0
            
            # Combined score
            combined_score = (jaccard_score * 0.4) + (keyword_density * 0.6)
            relevance_scores.append(combined_score)
        
        return relevance_scores
    
    def calculate_length_quality_score(self, chunks: List[str]) -> List[float]:
        """Calculate quality score based on chunk length and structure"""
        quality_scores = []
        
        for chunk in chunks:
            # Optimal length (500-1500 characters)
            length = len(chunk)
            if 500 <= length <= 1500:
                length_score = 1.0
            elif length < 500:
                length_score = length / 500
            else:
                length_score = max(0.3, 1500 / length)
            
            # Structure indicators (headers, bullet points, etc.)
            structure_indicators = ['-', '•', '#', '**', '1.', '2.', '3.']
            structure_score = min(1.0, sum(1 for indicator in structure_indicators if indicator in chunk) * 0.1)
            
            # Combined quality score
            quality_score = (length_score * 0.7) + (structure_score * 0.3)
            quality_scores.append(quality_score)
        
        return quality_scores
    
    def rerank_with_detailed_scores(self, query: str, chunks: List[str], metadata: List[Dict] = None) -> List[Dict]:
        """Rerank chunks with detailed scoring breakdown"""
        if not chunks:
            return []
        
        # Calculate different types of scores
        semantic_scores = self.calculate_semantic_similarity(query, chunks)
        keyword_scores = self.calculate_keyword_relevance(query, chunks)
        quality_scores = self.calculate_length_quality_score(chunks)
        
        # Combine scores with weights
        final_scores = []
        for i in range(len(chunks)):
            semantic_weight = 0.5
            keyword_weight = 0.3
            quality_weight = 0.2
            
            final_score = (
                semantic_scores[i] * semantic_weight +
                keyword_scores[i] * keyword_weight +
                quality_scores[i] * quality_weight
            )
            final_scores.append(final_score)
        
        # Create detailed results
        detailed_results = []
        for i, chunk in enumerate(chunks):
            result = {
                'text': chunk,
                'rank': i + 1,  # Will be updated after sorting
                'scores': {
                    'final_score': final_scores[i],
                    'semantic_similarity': semantic_scores[i],
                    'keyword_relevance': keyword_scores[i],
                    'content_quality': quality_scores[i]
                },
                'metadata': metadata[i] if metadata and i < len(metadata) else {}
            }
            detailed_results.append(result)
        
        # Sort by final score (descending)
        detailed_results.sort(key=lambda x: x['scores']['final_score'], reverse=True)
        
        # Update ranks
        for i, result in enumerate(detailed_results):
            result['rank'] = i + 1
            result['relevance_grade'] = self._get_relevance_grade(result['scores']['final_score'])
        
        return detailed_results
    
    def _get_relevance_grade(self, score: float) -> str:
        """Convert score to relevance grade"""
        if score >= 0.8:
            return "🟢 Sangat Relevan"
        elif score >= 0.6:
            return "🟡 Relevan"
        elif score >= 0.4:
            return "🟠 Cukup Relevan"
        else:
            return "🔴 Kurang Relevan"

# Global reranker instance
@st.cache_resource
def get_reranker():
    """Get cached reranker instance"""
    reranker = EnhancedReranker()
    if reranker.load_model():
        return reranker
    return None