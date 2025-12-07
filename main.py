import os
import sys
import numpy as np
import pandas as pd

print("=" * 80)
print("SCRIPT STARTED")
print("=" * 80)
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print()

try:
    print("Importing modules...")
    from src.data_preprocessing import DataPreprocessor
    print("   DataPreprocessor imported")
    
    from src.model import NeuralNetworkModel
    print("   NeuralNetworkModel imported")
    
    from src.evaluation import ModelEvaluator
    print("   ModelEvaluator imported")
    
    from src.hyperparameter_tuning import HyperparameterTuner
    print("   HyperparameterTuner imported")
    
    from src.visualization import Visualizer
    print("   Visualizer imported")
    
except ImportError as e:
    print(f"\n Import Error: {e}")
    import traceback
    traceback.print_exc()
    input("\nPress Enter to exit...")
    sys.exit(1)

np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

print("\n All imports successful")
print()


def main():
    print("=" * 80)
    print("MAIN FUNCTION STARTED")
    print("=" * 80)
    print()
    
    DATA_DIR = 'data/'
    TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
    TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
    EVENT_PATH = os.path.join(DATA_DIR, 'event_type.csv')
    LOG_PATH = os.path.join(DATA_DIR, 'log_feature.csv')
    RESOURCE_PATH = os.path.join(DATA_DIR, 'resource_type.csv')
    SEVERITY_PATH = os.path.join(DATA_DIR, 'severity_type.csv')
    
    print("Checking data files...")
    for path in [TRAIN_PATH, TEST_PATH, EVENT_PATH, LOG_PATH, RESOURCE_PATH, SEVERITY_PATH]:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"   {path} ({size:,} bytes)")
        else:
            print(f"   {path} NOT FOUND")
            input("\nPress Enter to exit...")
            return
    
    print()
    
    # Model hyperparameters
    LAYERS_CONFIG = [128, 64, 32]
    ACTIVATION = 'relu'
    DROPOUT_RATE = 0.3
    OPTIMIZER = 'adam'
    LEARNING_RATE = 0.001
    EPOCHS = 100
    BATCH_SIZE = 32
    
    print("\n" + "=" * 80)
    print(" TELSTRA NETWORK DISRUPTIONS - NEURAL NETWORK PIPELINE")
    print("=" * 80)
    
    # ==================== 1. DATA PREPROCESSING ====================
    print("\n[STEP 1] Initializing Data Preprocessor...")
    preprocessor = DataPreprocessor()
    
    print("\n[STEP 2] Loading and merging data...")
    df = preprocessor.load_and_merge_data(
        TRAIN_PATH, TEST_PATH, EVENT_PATH, 
        LOG_PATH, RESOURCE_PATH, SEVERITY_PATH
    )
    print(f"  - Data loaded: {df.shape}")
    
    print("\n[STEP 3] Cleaning data...")
    df = preprocessor.clean_data(df)
    print(f"  - Data cleaned: {df.shape}")
    
    print("\n[STEP 4] Feature engineering...")
    df = preprocessor.feature_engineering(df)
    print(f"  - Features engineered: {df.shape}")
    
    print("\n[STEP 5] Preparing data splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_data(
        df, target_col='fault_severity', test_size=0.2, val_size=0.2
    )
    
    print(f"  - Training set: {X_train.shape}")
    print(f"  - Validation set: {X_val.shape}")
    print(f"  - Test set: {X_test.shape}")
    
    # ==================== 2. MODEL DESIGN & TRAINING ====================
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    print(f"\n[STEP 6] Building model...")
    print(f"  - Input dimension: {input_dim}")
    print(f"  - Number of classes: {num_classes}")
    
    model = NeuralNetworkModel(input_dim, num_classes)
    model.build_model(
        layers_config=LAYERS_CONFIG,
        activation=ACTIVATION,
        dropout_rate=DROPOUT_RATE,
        optimizer=OPTIMIZER,
        learning_rate=LEARNING_RATE
    )
    
    print("\n[STEP 7] Training model...")
    history = model.train_model(
        X_train, y_train,
        X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # ==================== 3. MODEL EVALUATION ====================
    print("\n[STEP 8] Evaluating model...")
    evaluator = ModelEvaluator(model, class_names=['No Fault', 'Few Faults', 'Many Faults'])
    evaluation_results = evaluator.evaluate(X_test, y_test)
    
    # ==================== 4. HYPERPARAMETER TUNING ====================
    print("\n[STEP 9] Hyperparameter tuning (optional)")
    print("  - Set ENABLE_TUNING = True to run hyperparameter tuning")
    print("  - Warning: This can take 1-3 hours!")
    
    ENABLE_TUNING = True  
    
    if ENABLE_TUNING:
        print("\n" + "=" * 80)
        print("4. HYPERPARAMETER TUNING")
        print("=" * 80)
        
        param_grid = {
            'layers_config': [[64, 32], [128, 64], [128, 64, 32]],
            'activation': ['relu', 'tanh'],
            'dropout_rate': [0.2, 0.3, 0.4],
            'optimizer': ['adam', 'rmsprop'],
            'learning_rate': [0.001, 0.0001]
        }
        
        print("\n[Step 4.1] Initializing Hyperparameter Tuner...")
        tuner = HyperparameterTuner(input_dim, num_classes)
        
        print("\n[Step 4.2] Running Grid Search...")
        print(f"  - Testing {3 * 2 * 3 * 2 * 2} = 72 configurations")
        print(f"  - This will take approximately 1-2 hours")
        print()
        
        tuning_results = tuner.grid_search(
            X_train, y_train, 
            X_val, y_val,
            param_grid, 
            epochs=50,  
            batch_size=32
        )
        
        best_params = tuner.get_best_params()
        
        print("\n[Step 4.3] Best Hyperparameters Found:")
        for key, value in best_params.items():
            print(f"  - {key}: {value}")
        
        os.makedirs('outputs/reports', exist_ok=True)
        tuner.save_results('outputs/reports/hyperparameter_tuning_results.csv')
        
        with open('outputs/reports/best_hyperparameters.txt', 'w') as f:
            f.write("Best Hyperparameters from Grid Search\n")
            f.write("=" * 50 + "\n\n")
            for key, value in best_params.items():
                f.write(f"{key}: {value}\n")
        
        print("\n[Step 4.4] Results saved to outputs/reports/")
        
        print("\n[Step 4.5] Analyzing hyperparameter effects...")
        analysis = tuner.analyze_hyperparameters()
        
        print("\n" + "=" * 80)
        print("HYPERPARAMETER TUNING COMPLETE!")
        print("=" * 80)
        print("\nYou can now:")
        print("1. Check results: outputs/reports/hyperparameter_tuning_results.csv")
        print("2. Use best parameters by updating LAYERS_CONFIG, ACTIVATION, etc. in main.py")
        print("3. Retrain model with optimized hyperparameters")
        print()
        
        input("Press Enter to continue to predictions...")
    
    else:
        print("  - Hyperparameter tuning skipped (ENABLE_TUNING = False)")
        print("  - Using default hyperparameters")
    
    # ==================== 5. PREDICTION ON UNSEEN DATA ====================
    print("\n" + "=" * 80)
    print("5. PREDICTION ON UNSEEN DATA")
    print("=" * 80)
    print("\n[Step 5.1] Making predictions on test set...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print("\n[Step 5.2] Sample Predictions:")
    sample_size = min(10, len(X_test))
    for i in range(sample_size):
        print(f"  Sample {i+1}:")
        print(f"    True Label: {y_test.iloc[i]}")
        print(f"    Predicted Label: {y_pred[i]}")
        print(f"    Prediction Probabilities: {y_pred_proba[i]}")
    
    # ==================== 6. VISUALIZATION & REPORTING ====================
    print("\n[STEP 10] Creating visualizations...")
    visualizer = Visualizer()
    
    print("  - Plotting training history...")
    visualizer.plot_training_history(history, save_path='training_history.png')
    
    print("  - Plotting confusion matrix...")
    visualizer.plot_confusion_matrix(
        y_test, y_pred,
        class_names=['No Fault', 'Few Faults', 'Many Faults'],
        save_path='confusion_matrix.png'
    )
    
    print("  - Plotting class distribution...")
    visualizer.plot_class_distribution(
        y_train, y_val, y_test,
        save_path='class_distribution.png'
    )
    
    print("  - Plotting prediction distribution...")
    visualizer.plot_prediction_distribution(
        y_test, y_pred,
        save_path='prediction_distribution.png'
    )
    
    print("  - Plotting metrics comparison...")
    metrics_dict = {
        'Accuracy': evaluation_results['accuracy'],
        'Precision': evaluation_results['precision'],
        'Recall': evaluation_results['recall'],
        'F1-Score': evaluation_results['f1_score']
    }
    visualizer.plot_metrics_comparison(metrics_dict, save_path='metrics_comparison.png')
    
    # ==================== SAVE MODEL ====================
    print("\n[STEP 11] Saving model...")
    model.save_model('telstra_nn_model.keras')
    
    # ==================== FINAL SUMMARY ====================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nFinal Results:")
    print(f"  - Test Accuracy:  {evaluation_results['accuracy']:.4f}")
    print(f"  - Test Precision: {evaluation_results['precision']:.4f}")
    print(f"  - Test Recall:    {evaluation_results['recall']:.4f}")
    print(f"  - Test F1-Score:  {evaluation_results['f1_score']:.4f}")
    print("\nAll visualizations saved to current directory.")
    print("Model saved as 'telstra_nn_model.keras'")
    print("=" * 80)


if __name__ == "__main__":
    try:
        print("\nCalling main()...")
        main()
        print("\nmain() completed")
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR IN MAIN FUNCTION")
        print("=" * 80)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
    finally:
        input("\nPress Enter to exit...")