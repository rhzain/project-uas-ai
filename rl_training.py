import torch
from typing import List, Dict
import json
from datetime import datetime

class RLTrainer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.training_history = []
        
        # Simplified RL trainer - in practice you'd use libraries like TRL
        print(f"Initialized RL trainer for model: {model_name}")
    
    def compute_reward(self, question: str, answer: str, context: str, feedback_score: int) -> float:
        """Compute reward based on user feedback"""
        # Normalize feedback score to [-1, 1] range
        reward = (feedback_score - 3) / 2  # Assuming 1-5 scale
        
        # Add additional reward components
        relevance_reward = self.compute_relevance_reward(answer, context)
        coherence_reward = self.compute_coherence_reward(answer)
        
        total_reward = reward + 0.3 * relevance_reward + 0.2 * coherence_reward
        return total_reward
    
    def compute_relevance_reward(self, answer: str, context: str) -> float:
        """Simple relevance reward based on keyword overlap"""
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        if len(answer_words) == 0:
            return 0.0
        
        overlap = len(answer_words.intersection(context_words))
        return overlap / len(answer_words)
    
    def compute_coherence_reward(self, answer: str) -> float:
        """Simple coherence reward based on answer length and structure"""
        if len(answer.strip()) < 10:
            return 0.0
        
        # Simple heuristics
        sentences = answer.split('.')
        if len(sentences) > 1:
            return 0.5
        return 0.2
    
    def train_step(self, questions: List[str], contexts: List[str], answers: List[str], rewards: List[float]) -> Dict:
        """Single training step with RL"""
        # Simplified training step - in practice would update model weights
        
        training_data = {
            'timestamp': datetime.now().isoformat(),
            'batch_size': len(questions),
            'avg_reward': sum(rewards) / len(rewards) if rewards else 0,
            'max_reward': max(rewards) if rewards else 0,
            'min_reward': min(rewards) if rewards else 0
        }
        
        self.training_history.append(training_data)
        
        print(f"RL Training step completed:")
        print(f"  Batch size: {training_data['batch_size']}")
        print(f"  Average reward: {training_data['avg_reward']:.3f}")
        
        return training_data
    
    def get_training_stats(self) -> Dict:
        """Get training statistics"""
        if not self.training_history:
            return {"message": "No training steps completed yet"}
        
        avg_rewards = [step['avg_reward'] for step in self.training_history]
        
        return {
            "total_steps": len(self.training_history),
            "overall_avg_reward": sum(avg_rewards) / len(avg_rewards),
            "recent_avg_reward": avg_rewards[-1] if avg_rewards else 0,
            "training_trend": "improving" if len(avg_rewards) > 1 and avg_rewards[-1] > avg_rewards[0] else "stable"
        }

def create_rl_batches(rl_data: List[Dict], batch_size: int = 4) -> List[List[Dict]]:
    """Create batches for RL training"""
    batches = []
    for i in range(0, len(rl_data), batch_size):
        batch = rl_data[i:i + batch_size]
        batches.append(batch)
    return batches