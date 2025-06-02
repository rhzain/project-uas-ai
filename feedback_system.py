import json
import os
from datetime import datetime
from typing import List, Dict, Any

class FeedbackCollector:
    def __init__(self, feedback_file="user_feedback.json"):
        self.feedback_file = feedback_file
        self.feedback_db = self.load_existing_feedback()
    
    def load_existing_feedback(self) -> List[Dict]:
        """Load existing feedback from file"""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_feedback(self):
        """Save feedback to file"""
        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            json.dump(self.feedback_db, f, indent=2, ensure_ascii=False)
    
    def collect_answer_feedback(self, question: str, answer: str, context: str, rating: int, explanation: str = None) -> Dict:
        """Collect user feedback on answer quality"""
        feedback = {
            'id': len(self.feedback_db) + 1,
            'question': question,
            'answer': answer,
            'context': context,
            'rating': rating,  # 1-5 scale
            'explanation': explanation,
            'timestamp': datetime.now().isoformat(),
            'feedback_type': 'answer_quality'
        }
        self.feedback_db.append(feedback)
        self.save_feedback()
        return feedback
    
    def collect_retrieval_feedback(self, question: str, retrieved_chunks: List[str], relevance_scores: List[int]) -> Dict:
        """Collect feedback on retrieval quality"""
        feedback = {
            'id': len(self.feedback_db) + 1,
            'question': question,
            'retrieved_chunks': retrieved_chunks,
            'relevance_scores': relevance_scores,  # User-rated relevance for each chunk
            'timestamp': datetime.now().isoformat(),
            'feedback_type': 'retrieval_quality'
        }
        self.feedback_db.append(feedback)
        self.save_feedback()
        return feedback
    
    def prepare_rl_dataset(self) -> List[Dict]:
        """Prepare reward dataset for RL training"""
        rl_data = []
        
        for feedback in self.feedback_db:
            if feedback['feedback_type'] == 'answer_quality':
                # Normalize rating to [-1, 1] range
                reward = (feedback['rating'] - 3) / 2
                
                rl_data.append({
                    'state': {
                        'question': feedback['question'],
                        'context': feedback['context']
                    },
                    'action': feedback['answer'],
                    'reward': reward,
                    'next_state': None  # Terminal state for QA
                })
        
        return rl_data
    
    def prepare_embedding_training_data(self) -> List[Dict]:
        """Prepare triplet data for embedding fine-tuning"""
        triplets = []
        
        # Get positive examples (rating >= 4)
        positive_feedback = [f for f in self.feedback_db if f.get('rating', 0) >= 4]
        
        for feedback in positive_feedback:
            triplets.append({
                'anchor': feedback['question'],
                'positive': feedback['context'],
                'negative': self.get_negative_context(feedback['question'])
            })
        
        return triplets
    
    def get_negative_context(self, question: str) -> str:
        """Get a random context that's not relevant to the question"""
        # Simple implementation - in practice, you'd want more sophisticated negative sampling
        negative_contexts = [
            "This is about a completely different topic.",
            "Unrelated information about other subjects.",
            "Random text that doesn't answer the question."
        ]
        import random
        return random.choice(negative_contexts)
    
    def get_feedback_stats(self) -> Dict:
        """Get statistics about collected feedback"""
        if not self.feedback_db:
            return {"total_feedback": 0}
        
        ratings = [f['rating'] for f in self.feedback_db if f.get('rating')]
        
        return {
            "total_feedback": len(self.feedback_db),
            "avg_rating": sum(ratings) / len(ratings) if ratings else 0,
            "rating_distribution": {i: ratings.count(i) for i in range(1, 6)},
            "feedback_types": {ft: len([f for f in self.feedback_db if f.get('feedback_type') == ft]) 
                             for ft in set(f.get('feedback_type', 'unknown') for f in self.feedback_db)}
        }