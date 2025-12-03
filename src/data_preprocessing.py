import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.categorical_features = []
        self.numerical_features = []
        
    def load_and_merge_data(self, train_path, test_path, event_path, 
                           log_path, resource_path, severity_path, is_test=False):
        print("\n" + "=" * 80)
        print("1. DATA PREPROCESSING")
        print("=" * 80)
        print("\n[Step 1.1] Loading data files...")
        
        # Load main dataset
        if is_test:
            df = pd.read_csv(test_path)
            print(f"  - Test data loaded: {df.shape}")
        else:
            df = pd.read_csv(train_path)
            print(f"  - Train data loaded: {df.shape}")
        
        event_df = pd.read_csv(event_path)
        log_df = pd.read_csv(log_path)
        resource_df = pd.read_csv(resource_path)
        severity_df = pd.read_csv(severity_path)
        
        print(f"  - Event types: {event_df.shape}")
        print(f"  - Log features: {log_df.shape}")
        print(f"  - Resource types: {resource_df.shape}")
        print(f"  - Severity types: {severity_df.shape}")
        
        print("\n[Step 1.2] Merging additional features...")
        
        event_counts = event_df.groupby('id')['event_type'].agg(['count', 'nunique']).reset_index()
        event_counts.columns = ['id', 'event_count', 'event_unique']
        df = df.merge(event_counts, on='id', how='left')
        
        log_counts = log_df.groupby('id')['log_feature'].agg(['count', 'nunique']).reset_index()
        log_counts.columns = ['id', 'log_count', 'log_unique']
        df = df.merge(log_counts, on='id', how='left')
        
        resource_counts = resource_df.groupby('id')['resource_type'].agg(['count', 'nunique']).reset_index()
        resource_counts.columns = ['id', 'resource_count', 'resource_unique']
        df = df.merge(resource_counts, on='id', how='left')
        
        severity_counts = severity_df.groupby('id')['severity_type'].agg(['count', 'nunique']).reset_index()
        severity_counts.columns = ['id', 'severity_count', 'severity_unique']
        df = df.merge(severity_counts, on='id', how='left')
        
        event_mode = event_df.groupby('id')['event_type'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown').reset_index()
        event_mode.columns = ['id', 'event_type_mode']
        df = df.merge(event_mode, on='id', how='left')
        
        resource_mode = resource_df.groupby('id')['resource_type'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown').reset_index()
        resource_mode.columns = ['id', 'resource_type_mode']
        df = df.merge(resource_mode, on='id', how='left')
        
        severity_mode = severity_df.groupby('id')['severity_type'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown').reset_index()
        severity_mode.columns = ['id', 'severity_type_mode']
        df = df.merge(severity_mode, on='id', how='left')
        
        print(f"  - Merged data shape: {df.shape}")
        
        return df
    
    def clean_data(self, df):
        print("\n[Step 1.3] Data Cleaning...")
        
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"  - Missing values found:")
            print(missing[missing > 0])
            df = df.fillna(0)
            print(f"  - Missing values filled with 0")
        else:
            print(f"  - No missing values found")
        
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"  - Duplicates found: {duplicates}")
            df = df.drop_duplicates()
            print(f"  - Duplicates removed")
        else:
            print(f"  - No duplicates found")

        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in ['id', 'fault_severity']]
        
        outliers_removed = 0
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                outliers_removed += outliers
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        if outliers_removed > 0:
            print(f"  - Outliers capped: {outliers_removed} values")
        else:
            print(f"  - No significant outliers found")
        
        print(f"  - Final cleaned data shape: {df.shape}")
        
        return df
    
    def feature_engineering(self, df, is_test=False):
        print("\n[Step 1.4] Feature Engineering...")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col not in ['id']]
        
        print(f"  - Categorical features: {categorical_cols}")
        
        for col in categorical_cols:
            if col in df.columns:
                if not is_test:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        df[col] = df[col].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
        
        if 'event_count' in df.columns and 'log_count' in df.columns:
            df['event_log_ratio'] = df['event_count'] / (df['log_count'] + 1)
        
        if 'severity_count' in df.columns and 'resource_count' in df.columns:
            df['severity_resource_ratio'] = df['severity_count'] / (df['resource_count'] + 1)
        
        count_cols = [col for col in df.columns if 'count' in col]
        if len(count_cols) > 0:
            df['total_features'] = df[count_cols].sum(axis=1)
        
        unique_cols = [col for col in df.columns if 'unique' in col]
        if len(unique_cols) > 0:
            df['total_unique'] = df[unique_cols].sum(axis=1)
        
        print(f"  - Features after engineering: {df.shape[1]}")
        
        return df
    
    def scale_features(self, X_train, X_val=None, X_test=None):
        print("\n[Step 1.5] Feature Scaling...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        print(f"  - Training features scaled: {X_train_scaled.shape}")
        
        X_val_scaled = None
        X_test_scaled = None
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            print(f"  - Validation features scaled: {X_val_scaled.shape}")
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            print(f"  - Test features scaled: {X_test_scaled.shape}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def prepare_data(self, df, target_col='fault_severity', test_size=0.2, val_size=0.2):
        print("\n[Step 1.6] Preparing data for training...")
        
        X = df.drop([target_col, 'id'], axis=1, errors='ignore')
        y = df[target_col]
        
        self.feature_names = X.columns.tolist()
        
        print(f"  - Features: {X.shape[1]}")
        print(f"  - Target distribution:")
        print(y.value_counts().sort_index())
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=42, stratify=y_train_val
        )
        
        print(f"\n  - Training set: {X_train.shape}")
        print(f"  - Validation set: {X_val.shape}")
        print(f"  - Test set: {X_test.shape}")
        
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test
        )
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def get_feature_names(self):
        return self.feature_names