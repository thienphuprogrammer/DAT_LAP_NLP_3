from typing import List, Dict

class MetricsCalculator:
    """Metrics calculation utilities"""
    
    def calculate_bleu(self, references: List[str], 
                      hypotheses: List[str]) -> float:
        pass
    
    def calculate_classification_metrics(self, y_true, y_pred) -> Dict:
        # Output: accuracy, f1_score, precision, recall
        pass
    
    def calculate_rouge(self, references: List[str],
                       hypotheses: List[str]) -> Dict:
        pass