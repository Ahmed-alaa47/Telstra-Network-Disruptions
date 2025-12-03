import pickle
import os
from src.data_preprocessing import DataPreprocessor

def save_preprocessor():
    print("=" * 80)
    print("SAVING PREPROCESSOR FOR DEPLOYMENT")
    print("=" * 80)
    print()
    
    DATA_DIR = 'data/'
    TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
    TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
    EVENT_PATH = os.path.join(DATA_DIR, 'event_type.csv')
    LOG_PATH = os.path.join(DATA_DIR, 'log_feature.csv')
    RESOURCE_PATH = os.path.join(DATA_DIR, 'resource_type.csv')
    SEVERITY_PATH = os.path.join(DATA_DIR, 'severity_type.csv')
    
    os.makedirs('outputs/models', exist_ok=True)
    
    print("Initializing preprocessor...")
    preprocessor = DataPreprocessor()
    
    print("Loading and processing data...")
    df = preprocessor.load_and_merge_data(
        TRAIN_PATH, TEST_PATH, EVENT_PATH, 
        LOG_PATH, RESOURCE_PATH, SEVERITY_PATH
    )
    
    df = preprocessor.clean_data(df)
    df = preprocessor.feature_engineering(df)
    
    print("Fitting preprocessor...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_data(
        df, target_col='fault_severity', test_size=0.2, val_size=0.2
    )
    
    print("\nSaving preprocessor...")
    preprocessor_path = 'outputs/models/preprocessor.pkl'
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print(f"✓ Preprocessor saved to '{preprocessor_path}'")
    print()
    print("=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    print("\nYou can now run the API server:")
    print("  python run_api.py")
    print()

if __name__ == "__main__":
    try:
        save_preprocessor()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("\nPress Enter to exit...")