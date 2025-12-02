"""
Model Evaluation Module
Handles performance assessment and metrics calculation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)


class ModelEvaluator:
    """
    Evaluates model performance using various metrics
    """
    
    def __init__(self, model, class_names=None):
        """
        Initialize evaluator
        
        Parameters:
        -----------
        model : NeuralNetworkModel
            Trained model
        class_names : list, optional
            Names of classes
        """
        self.model = model
        self.class_names = class_names or ['0', '1', '2']
        self.evaluation_results = {}
        
    def evaluate(self, X_test, y_test):
        """
        Comprehensive model evaluation
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
            
        Returns:
        --------
        dict
            Dictionary containing all evaluation metrics
        """
        print("\n" + "=" * 80)
        print("3. MODEL EVALUATION")
        print("=" * 80)
        print("\n[Step 3.1] Evaluating model on test set...")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Store results
        self.evaluation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Print results
        print("\n[Step 3.2] Classification Metrics:")
        print(f"  - Accuracy:  {accuracy:.4f}")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall:    {recall:.4f}")
        print(f"  - F1-Score:  {f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        self.evaluation_results['confusion_matrix'] = cm
        
        print("\n[Step 3.3] Confusion Matrix:")
        print(cm)
        
        # Classification Report
        print("\n[Step 3.4] Detailed Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=self.class_names,
                                   digits=4))
        
        # Per-class metrics
        print("\n[Step 3.5] Per-Class Metrics:")
        for i, class_name in enumerate(self.class_names):
            class_precision = precision_score(y_test, y_pred, 
                                            labels=[i], average='macro')
            class_recall = recall_score(y_test, y_pred, 
                                       labels=[i], average='macro')
            class_f1 = f1_score(y_test, y_pred, 
                              labels=[i], average='macro')
            
            print(f"  Class {class_name}:")
            print(f"    - Precision: {class_precision:.4f}")
            print(f"    - Recall:    {class_recall:.4f}")
            print(f"    - F1-Score:  {class_f1:.4f}")
        
        return self.evaluation_results
    
    def get_results(self):
        """Return evaluation results"""
        return self.evaluation_results