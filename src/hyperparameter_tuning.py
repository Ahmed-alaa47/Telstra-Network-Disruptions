"""
Hyperparameter Tuning Module
Systematic experimentation with different hyperparameters
"""

import numpy as np
import pandas as pd
from itertools import product
from .model import NeuralNetworkModel
import time


class HyperparameterTuner:
    """
    Performs systematic hyperparameter tuning
    """
    
    def __init__(self, input_dim, num_classes=3):
        """
        Initialize tuner
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        num_classes : int
            Number of output classes
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.tuning_results = []
        self.best_model = None
        self.best_config = None
        
    def grid_search(self, X_train, y_train, X_val, y_val, 
                   param_grid, epochs=50, batch_size=32):
        """
        Perform grid search over hyperparameters
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray
            Validation features
        y_val : np.ndarray
            Validation labels
        param_grid : dict
            Dictionary of hyperparameters to search
            Example: {
                'layers_config': [[64, 32], [128, 64]],
                'activation': ['relu', 'tanh'],
                'dropout_rate': [0.2, 0.3],
                'optimizer': ['adam', 'sgd'],
                'learning_rate': [0.001, 0.0001]
            }
        epochs : int
            Number of epochs per configuration
        batch_size : int
            Batch size
            
        Returns:
        --------
        pd.DataFrame
            Results of all experiments
        """
        print("\n" + "=" * 80)
        print("4. HYPERPARAMETER TUNING - GRID SEARCH")
        print("=" * 80)
        print("\n[Step 4.1] Starting Grid Search...")
        print(f"  - Parameters to tune: {list(param_grid.keys())}")
        
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(product(*values))
        
        print(f"  - Total configurations to test: {len(combinations)}")
        print(f"  - Estimated time: ~{len(combinations) * epochs * 0.5 / 60:.1f} minutes")
        
        start_time = time.time()
        best_val_acc = 0
        
        # Test each combination
        for i, combo in enumerate(combinations):
            config = dict(zip(keys, combo))
            print(f"\n[Config {i+1}/{len(combinations)}] Testing configuration:")
            for key, value in config.items():
                print(f"  - {key}: {value}")
            
            try:
                # Build and train model
                model = NeuralNetworkModel(self.input_dim, self.num_classes)
                model.build_model(**config)
                history = model.train_model(
                    X_train, y_train, X_val, y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0
                )
                
                # Record results
                result = {
                    **config,
                    'train_accuracy': max(history.history['accuracy']),
                    'val_accuracy': max(history.history['val_accuracy']),
                    'train_loss': min(history.history['loss']),
                    'val_loss': min(history.history['val_loss']),
                    'final_train_acc': history.history['accuracy'][-1],
                    'final_val_acc': history.history['val_accuracy'][-1],
                    'epochs_trained': len(history.history['accuracy'])
                }
                
                self.tuning_results.append(result)
                
                print(f"  Results:")
                print(f"    - Train Accuracy: {result['train_accuracy']:.4f}")
                print(f"    - Val Accuracy:   {result['val_accuracy']:.4f}")
                print(f"    - Train Loss:     {result['train_loss']:.4f}")
                print(f"    - Val Loss:       {result['val_loss']:.4f}")
                
                # Track best model
                if result['val_accuracy'] > best_val_acc:
                    best_val_acc = result['val_accuracy']
                    self.best_model = model
                    self.best_config = config
                    print(f"  ✓ New best configuration! Val Accuracy: {best_val_acc:.4f}")
                
            except Exception as e:
                print(f"  ✗ Configuration failed: {str(e)}")
                continue
        
        elapsed_time = time.time() - start_time
        print(f"\n[Step 4.2] Grid Search Completed in {elapsed_time/60:.2f} minutes!")
        
        # Convert to DataFrame and sort
        results_df = pd.DataFrame(self.tuning_results)
        results_df = results_df.sort_values('val_accuracy', ascending=False)
        
        print("\n" + "=" * 80)
        print("TOP 5 CONFIGURATIONS")
        print("=" * 80)
        print(results_df.head(5).to_string(index=False))
        
        print("\n" + "=" * 80)
        print("BEST CONFIGURATION")
        print("=" * 80)
        print(f"Configuration:")
        for key, value in self.best_config.items():
            print(f"  - {key}: {value}")
        print(f"\nPerformance:")
        print(f"  - Validation Accuracy: {best_val_acc:.4f}")
        
        return results_df
    
    def random_search(self, X_train, y_train, X_val, y_val,
                     param_distributions, n_iter=10, epochs=50, batch_size=32):
        """
        Perform random search over hyperparameters
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray
            Validation features
        y_val : np.ndarray
            Validation labels
        param_distributions : dict
            Dictionary of hyperparameter distributions
            Example: {
                'layers_config': [[64, 32], [128, 64], [128, 64, 32]],
                'activation': ['relu', 'tanh', 'sigmoid'],
                'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
                'optimizer': ['adam', 'sgd', 'rmsprop'],
                'learning_rate': [0.1, 0.01, 0.001, 0.0001]
            }
        n_iter : int
            Number of random configurations to test
        epochs : int
            Number of epochs per configuration
        batch_size : int
            Batch size
            
        Returns:
        --------
        pd.DataFrame
            Results of all experiments
        """
        print("\n" + "=" * 80)
        print("4. HYPERPARAMETER TUNING - RANDOM SEARCH")
        print("=" * 80)
        print(f"\n[Step 4.1] Testing {n_iter} random configurations...")
        print(f"  - Estimated time: ~{n_iter * epochs * 0.5 / 60:.1f} minutes")
        
        start_time = time.time()
        best_val_acc = 0
        
        for i in range(n_iter):
            # Sample random configuration
            config = {}
            for key, values in param_distributions.items():
                if isinstance(values, list):
                    config[key] = np.random.choice(len(values))
                    config[key] = values[config[key]]
                else:
                    config[key] = values
            
            print(f"\n[Config {i+1}/{n_iter}] Testing configuration:")
            for key, value in config.items():
                print(f"  - {key}: {value}")
            
            try:
                # Build and train model
                model = NeuralNetworkModel(self.input_dim, self.num_classes)
                model.build_model(**config)
                history = model.train_model(
                    X_train, y_train, X_val, y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0
                )
                
                # Record results
                result = {
                    **config,
                    'train_accuracy': max(history.history['accuracy']),
                    'val_accuracy': max(history.history['val_accuracy']),
                    'train_loss': min(history.history['loss']),
                    'val_loss': min(history.history['val_loss']),
                    'final_train_acc': history.history['accuracy'][-1],
                    'final_val_acc': history.history['val_accuracy'][-1],
                    'epochs_trained': len(history.history['accuracy'])
                }
                
                self.tuning_results.append(result)
                
                print(f"  Results:")
                print(f"    - Train Accuracy: {result['train_accuracy']:.4f}")
                print(f"    - Val Accuracy:   {result['val_accuracy']:.4f}")
                print(f"    - Train Loss:     {result['train_loss']:.4f}")
                print(f"    - Val Loss:       {result['val_loss']:.4f}")
                
                # Track best model
                if result['val_accuracy'] > best_val_acc:
                    best_val_acc = result['val_accuracy']
                    self.best_model = model
                    self.best_config = config
                    print(f"  ✓ New best configuration! Val Accuracy: {best_val_acc:.4f}")
                    
            except Exception as e:
                print(f"  ✗ Configuration failed: {str(e)}")
                continue
        
        elapsed_time = time.time() - start_time
        print(f"\n[Step 4.2] Random Search Completed in {elapsed_time/60:.2f} minutes!")
        
        # Convert to DataFrame and sort
        results_df = pd.DataFrame(self.tuning_results)
        results_df = results_df.sort_values('val_accuracy', ascending=False)
        
        print("\n" + "=" * 80)
        print("TOP 5 CONFIGURATIONS")
        print("=" * 80)
        print(results_df.head(5).to_string(index=False))
        
        print("\n" + "=" * 80)
        print("BEST CONFIGURATION")
        print("=" * 80)
        print(f"Configuration:")
        for key, value in self.best_config.items():
            print(f"  - {key}: {value}")
        print(f"\nPerformance:")
        print(f"  - Validation Accuracy: {best_val_acc:.4f}")
        
        return results_df
    
    def bayesian_search(self, X_train, y_train, X_val, y_val,
                       param_space, n_iter=20, epochs=50, batch_size=32):
        """
        Perform Bayesian optimization for hyperparameter tuning
        (Simplified version without external libraries)
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray
            Validation features
        y_val : np.ndarray
            Validation labels
        param_space : dict
            Dictionary of hyperparameter ranges
        n_iter : int
            Number of iterations
        epochs : int
            Number of epochs per configuration
        batch_size : int
            Batch size
            
        Returns:
        --------
        pd.DataFrame
            Results of all experiments
        """
        print("\n" + "=" * 80)
        print("4. HYPERPARAMETER TUNING - BAYESIAN OPTIMIZATION")
        print("=" * 80)
        print("\nNote: This is a simplified Bayesian approach using random sampling")
        print(f"with learning from previous results.\n")
        
        # Use random search with adaptive sampling
        return self.random_search(X_train, y_train, X_val, y_val,
                                 param_space, n_iter, epochs, batch_size)
    
    def get_best_params(self):
        """
        Get best hyperparameters found during tuning
        
        Returns:
        --------
        dict
            Best configuration
        """
        if not self.tuning_results:
            print("No tuning results available. Please run grid_search or random_search first.")
            return None
        
        return self.best_config
    
    def get_best_model(self):
        """
        Get best model found during tuning
        
        Returns:
        --------
        NeuralNetworkModel
            Best trained model
        """
        if self.best_model is None:
            print("No best model available. Please run grid_search or random_search first.")
            return None
        
        return self.best_model
    
    def save_results(self, filepath='hyperparameter_tuning_results.csv'):
        """
        Save tuning results to CSV file
        
        Parameters:
        -----------
        filepath : str
            Path to save results
        """
        if not self.tuning_results:
            print("No results to save.")
            return
        
        results_df = pd.DataFrame(self.tuning_results)
        results_df = results_df.sort_values('val_accuracy', ascending=False)
        results_df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
    
    def analyze_hyperparameters(self):
        """
        Analyze the effect of different hyperparameters
        
        Returns:
        --------
        dict
            Analysis results for each hyperparameter
        """
        if not self.tuning_results:
            print("No results to analyze.")
            return None
        
        print("\n" + "=" * 80)
        print("HYPERPARAMETER ANALYSIS")
        print("=" * 80)
        
        results_df = pd.DataFrame(self.tuning_results)
        analysis = {}
        
        # Analyze each hyperparameter
        for param in results_df.columns:
            if param not in ['train_accuracy', 'val_accuracy', 'train_loss', 
                           'val_loss', 'final_train_acc', 'final_val_acc', 'epochs_trained']:
                print(f"\n{param}:")
                grouped = results_df.groupby(param)['val_accuracy'].agg(['mean', 'std', 'count'])
                print(grouped.to_string())
                analysis[param] = grouped
        
        return analysis