"""
Visualization Module
Handles all plotting and visualization tasks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


class Visualizer:
    """
    Creates various visualizations for model analysis
    """
    
    def __init__(self, style='default'):
        """Initialize visualizer with style"""
        plt.style.use(style)
        sns.set_palette("husl")
        self.fig_size = (12, 6)
        
    def plot_training_history(self, history, save_path=None):
        """
        Plot training and validation loss/accuracy curves
        
        Parameters:
        -----------
        history : keras.callbacks.History
            Training history object
        save_path : str, optional
            Path to save figure
        """
        print("\n" + "=" * 80)
        print("6. VISUALIZATION & REPORTING")
        print("=" * 80)
        print("\n[Step 6.1] Plotting training history...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(history.history['accuracy'], label='Training Accuracy', 
                    linewidth=2, marker='o', markersize=4)
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', 
                    linewidth=2, marker='s', markersize=4)
        axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(loc='lower right', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[1].plot(history.history['loss'], label='Training Loss', 
                    linewidth=2, marker='o', markersize=4)
        axes[1].plot(history.history['val_loss'], label='Validation Loss', 
                    linewidth=2, marker='s', markersize=4)
        axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(loc='upper right', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  - Saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, 
                             normalize=False, save_path=None):
        """
        Plot confusion matrix
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        class_names : list, optional
            Names of classes
        normalize : bool
            Whether to normalize the confusion matrix
        save_path : str, optional
            Path to save figure
        """
        print("\n[Step 6.2] Plotting confusion matrix...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=class_names or ['0', '1', '2'],
                   yticklabels=class_names or ['0', '1', '2'],
                   cbar_kws={'label': 'Percentage' if normalize else 'Count'},
                   linewidths=1, linecolor='gray')
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  - Saved to {save_path}")
        
        plt.show()
    
    def plot_class_distribution(self, y_train, y_val, y_test, save_path=None):
        """
        Plot class distribution across splits
        
        Parameters:
        -----------
        y_train : array-like
            Training labels
        y_val : array-like
            Validation labels
        y_test : array-like
            Test labels
        save_path : str, optional
            Path to save figure
        """
        print("\n[Step 6.3] Plotting class distribution...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Training set
        train_counts = pd.Series(y_train).value_counts().sort_index()
        axes[0].bar(train_counts.index, train_counts.values, 
                   color='skyblue', edgecolor='black', linewidth=1.5)
        axes[0].set_title('Training Set Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Fault Severity', fontsize=11)
        axes[0].set_ylabel('Count', fontsize=11)
        axes[0].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(train_counts.values):
            axes[0].text(i, v + max(train_counts.values)*0.02, str(v), 
                        ha='center', va='bottom', fontweight='bold')
        
        # Validation set
        val_counts = pd.Series(y_val).value_counts().sort_index()
        axes[1].bar(val_counts.index, val_counts.values, 
                   color='lightgreen', edgecolor='black', linewidth=1.5)
        axes[1].set_title('Validation Set Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Fault Severity', fontsize=11)
        axes[1].set_ylabel('Count', fontsize=11)
        axes[1].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(val_counts.values):
            axes[1].text(i, v + max(val_counts.values)*0.02, str(v), 
                        ha='center', va='bottom', fontweight='bold')
        
        # Test set
        test_counts = pd.Series(y_test).value_counts().sort_index()
        axes[2].bar(test_counts.index, test_counts.values, 
                   color='salmon', edgecolor='black', linewidth=1.5)
        axes[2].set_title('Test Set Distribution', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Fault Severity', fontsize=11)
        axes[2].set_ylabel('Count', fontsize=11)
        axes[2].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(test_counts.values):
            axes[2].text(i, v + max(test_counts.values)*0.02, str(v), 
                        ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  - Saved to {save_path}")
        
        plt.show()
    
    def plot_prediction_distribution(self, y_true, y_pred, save_path=None):
        """
        Plot comparison of true vs predicted distributions
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        save_path : str, optional
            Path to save figure
        """
        print("\n[Step 6.4] Plotting prediction distribution...")
        
        true_counts = pd.Series(y_true).value_counts().sort_index()
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        
        # Ensure both have same indices
        all_classes = sorted(set(list(true_counts.index) + list(pred_counts.index)))
        true_counts = true_counts.reindex(all_classes, fill_value=0)
        pred_counts = pred_counts.reindex(all_classes, fill_value=0)
        
        x = np.arange(len(all_classes))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        bars1 = plt.bar(x - width/2, true_counts.values, width, label='True', 
                       color='steelblue', edgecolor='black', linewidth=1.5)
        bars2 = plt.bar(x + width/2, pred_counts.values, width, label='Predicted', 
                       color='orange', edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xlabel('Fault Severity', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('True vs Predicted Distribution', fontsize=14, fontweight='bold')
        plt.xticks(x, all_classes)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  - Saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, y_true, y_pred_proba, num_classes=3, save_path=None):
        """
        Plot ROC curves for multi-class classification
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Prediction probabilities
        num_classes : int
            Number of classes
        save_path : str, optional
            Path to save figure
        """
        print("\n[Step 6.5] Plotting ROC curves...")
        
        # Binarize the labels
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green']
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                    label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-Class Classification', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  - Saved to {save_path}")
        
        plt.show()
    
    def plot_hyperparameter_comparison(self, results_df, param_name, 
                                      metric='val_accuracy', save_path=None):
        """
        Plot comparison of hyperparameter effects
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Results from hyperparameter tuning
        param_name : str
            Name of parameter to visualize
        metric : str
            Metric to plot ('val_accuracy', 'val_loss', etc.)
        save_path : str, optional
            Path to save figure
        """
        print(f"\n[Step 6.6] Plotting {param_name} comparison...")
        
        if param_name not in results_df.columns:
            print(f"  - Parameter '{param_name}' not found in results")
            return
        
        # Handle different types of parameters
        if results_df[param_name].dtype == 'object':
            # For categorical parameters (e.g., optimizer, activation)
            grouped = results_df.groupby(param_name)[metric].agg(['mean', 'std']).reset_index()
            
            plt.figure(figsize=(10, 6))
            x_pos = np.arange(len(grouped))
            plt.bar(x_pos, grouped['mean'], yerr=grouped['std'], 
                   capsize=5, color='skyblue', edgecolor='black', linewidth=1.5)
            plt.xticks(x_pos, grouped[param_name], rotation=45, ha='right')
            plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
            plt.xlabel(param_name.replace('_', ' ').title(), fontsize=12)
            plt.title(f'{metric.replace("_", " ").title()} vs {param_name.replace("_", " ").title()}',
                     fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
        else:
            # For numerical parameters (e.g., learning_rate, dropout_rate)
            grouped = results_df.groupby(param_name)[metric].agg(['mean', 'std']).reset_index()
            
            plt.figure(figsize=(10, 6))
            plt.errorbar(grouped[param_name], grouped['mean'], yerr=grouped['std'],
                        marker='o', linewidth=2, markersize=8, capsize=5)
            plt.xlabel(param_name.replace('_', ' ').title(), fontsize=12)
            plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
            plt.title(f'{metric.replace("_", " ").title()} vs {param_name.replace("_", " ").title()}',
                     fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  - Saved to {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, metrics_dict, save_path=None):
        """
        Plot comparison of different metrics
        
        Parameters:
        -----------
        metrics_dict : dict
            Dictionary of metrics {'metric_name': value}
        save_path : str, optional
            Path to save figure
        """
        print("\n[Step 6.7] Plotting metrics comparison...")
        
        metrics = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        
        plt.figure(figsize=(10, 6))
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        bars = plt.bar(metrics, values, color=colors[:len(metrics)],
                      edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
        plt.ylim(0, min(1.1, max(values) * 1.2))
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  - Saved to {save_path}")
        
        plt.show()
    
    def plot_learning_curves(self, train_sizes, train_scores, val_scores, save_path=None):
        """
        Plot learning curves to diagnose bias/variance
        
        Parameters:
        -----------
        train_sizes : array-like
            Training set sizes
        train_scores : array-like
            Training scores
        val_scores : array-like
            Validation scores
        save_path : str, optional
            Path to save figure
        """
        print("\n[Step 6.8] Plotting learning curves...")
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores, 'o-', linewidth=2, 
                label='Training Score', markersize=8)
        plt.plot(train_sizes, val_scores, 's-', linewidth=2, 
                label='Validation Score', markersize=8)
        
        plt.xlabel('Training Set Size', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Learning Curves', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  - Saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_names, importances, top_n=20, save_path=None):
        """
        Plot feature importance
        
        Parameters:
        -----------
        feature_names : list
            List of feature names
        importances : array-like
            Feature importance scores
        top_n : int
            Number of top features to display
        save_path : str, optional
            Path to save figure
        """
        print(f"\n[Step 6.9] Plotting top {top_n} feature importances...")
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_importances, 
                color='steelblue', edgecolor='black', linewidth=1.5)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  - Saved to {save_path}")
        
        plt.show()
    
    def plot_prediction_errors(self, y_true, y_pred, save_path=None):
        """
        Plot prediction errors analysis
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        save_path : str, optional
            Path to save figure
        """
        print("\n[Step 6.10] Plotting prediction errors...")
        
        errors = y_pred != y_true
        error_indices = np.where(errors)[0]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Error distribution by true class
        error_by_class = {}
        for cls in np.unique(y_true):
            cls_mask = y_true == cls
            cls_errors = np.sum(errors[cls_mask])
            cls_total = np.sum(cls_mask)
            error_by_class[cls] = cls_errors / cls_total if cls_total > 0 else 0
        
        axes[0].bar(error_by_class.keys(), error_by_class.values(), 
                   color='coral', edgecolor='black', linewidth=1.5)
        axes[0].set_title('Error Rate by True Class', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('True Class', fontsize=11)
        axes[0].set_ylabel('Error Rate', fontsize=11)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Misclassification pattern
        misclass_matrix = confusion_matrix(y_true[errors], y_pred[errors])
        sns.heatmap(misclass_matrix, annot=True, fmt='d', cmap='Reds', 
                   ax=axes[1], cbar_kws={'label': 'Count'})
        axes[1].set_title('Misclassification Pattern', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Predicted Label', fontsize=11)
        axes[1].set_ylabel('True Label', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  - Saved to {save_path}")
        
        plt.show()
    
    def create_comprehensive_report(self, history, y_true, y_pred, y_pred_proba,
                                   metrics_dict, class_names=None, save_dir='./'):
        """
        Create a comprehensive visualization report
        
        Parameters:
        -----------
        history : keras.callbacks.History
            Training history
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_pred_proba : array-like
            Prediction probabilities
        metrics_dict : dict
            Dictionary of evaluation metrics
        class_names : list, optional
            Names of classes
        save_dir : str
            Directory to save all plots
        """
        print("\n" + "=" * 80)
        print("CREATING COMPREHENSIVE VISUALIZATION REPORT")
        print("=" * 80)
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot all visualizations
        self.plot_training_history(history, 
                                   save_path=os.path.join(save_dir, 'training_history.png'))
        
        self.plot_confusion_matrix(y_true, y_pred, class_names=class_names,
                                  save_path=os.path.join(save_dir, 'confusion_matrix.png'))
        
        self.plot_prediction_distribution(y_true, y_pred,
                                         save_path=os.path.join(save_dir, 'prediction_distribution.png'))
        
        self.plot_roc_curves(y_true, y_pred_proba,
                           save_path=os.path.join(save_dir, 'roc_curves.png'))
        
        self.plot_metrics_comparison(metrics_dict,
                                    save_path=os.path.join(save_dir, 'metrics_comparison.png'))
        
        self.plot_prediction_errors(y_true, y_pred,
                                   save_path=os.path.join(save_dir, 'prediction_errors.png'))
        
        print(f"\n✓ All visualizations saved to {save_dir}")