from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import torch
from typing import List, Dict
import os

class EmbeddingFineTuner:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
    def prepare_training_examples(self, triplet_data: List[Dict]) -> List[InputExample]:
        """Convert triplet data to InputExamples"""
        examples = []
        
        for triplet in triplet_data:
            # Positive example
            examples.append(InputExample(
                texts=[triplet['anchor'], triplet['positive']], 
                label=1.0
            ))
            
            # Negative example
            examples.append(InputExample(
                texts=[triplet['anchor'], triplet['negative']], 
                label=0.0
            ))
        
        return examples
    
    def fine_tune(self, training_examples: List[InputExample], output_path: str, epochs: int = 1):
        """Fine-tune embedding model"""
        if not training_examples:
            print("No training examples provided")
            return self.model
        
        print(f"Fine-tuning with {len(training_examples)} examples...")
        
        train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=16)
        
        # Use CosineSimilarityLoss for similarity learning
        train_loss = losses.CosineSimilarityLoss(model=self.model)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Training
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=min(100, len(train_dataloader)),
            output_path=output_path,
            save_best_model=True,
            show_progress_bar=True
        )
        
        print(f"Fine-tuning completed. Model saved to {output_path}")
        return self.model
    
    def evaluate_model(self, test_examples: List[InputExample]) -> Dict:
        """Evaluate the fine-tuned model"""
        if not test_examples:
            return {"error": "No test examples provided"}
        
        # Create evaluator
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            test_examples, name='test_evaluation'
        )
        
        # Evaluate
        score = evaluator(self.model)
        return {"similarity_score": score}