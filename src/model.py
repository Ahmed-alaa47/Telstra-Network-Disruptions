"""
Neural Network Model Module
Handles model architecture, training, and configuration
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import callbacks


class NeuralNetworkModel:
    """
    Neural Network model for multi-class classification
    Supports experimentation with different architectures and hyperparameters
    """
    
    def __init__(self, input_dim, num_classes=3):
        """
        Initialize Neural Network model
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        num_classes : int
            Number of output classes
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self, 
                   layers_config=[128, 64, 32],
                   activation='relu',
                   dropout_rate=0.3,
                   optimizer='adam',
                   learning_rate=0.001):
        """
        Build neural network architecture
        
        Parameters:
        -----------
        layers_config : list
            List of neurons for each hidden layer
        activation : str
            Activation function ('relu', 'tanh', 'sigmoid')
        dropout_rate : float
            Dropout rate for regularization
        optimizer : str
            Optimizer ('adam', 'sgd', 'rmsprop')
        learning_rate : float
            Learning rate for optimizer
            
        Returns:
        --------
        keras.Model
            Compiled model
        """
        print("\n" + "=" * 80)
        print("2. MODEL DESIGN & TRAINING")
        print("=" * 80)
        print("\n[Step 2.1] Building Neural Network Model...")
        print(f"  - Input dimension: {self.input_dim}")
        print(f"  - Output classes: {self.num_classes}")
        print(f"  - Architecture: {layers_config}")
        print(f"  - Activation: {activation}")
        print(f"  - Dropout rate: {dropout_rate}")
        print(f"  - Optimizer: {optimizer}")
        print(f"  - Learning rate: {learning_rate}")
        
        # Input layer
        inputs = tf.keras.Input(shape=(self.input_dim,))
        x = inputs
        
        # Hidden layers
        for i, units in enumerate(layers_config):
            x = layers.Dense(units, activation=activation, 
                           name=f'hidden_{i+1}')(x)
            x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
            x = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', 
                             name='output')(x)
        
        # Create model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='TelstraNN')
        
        # Select optimizer
        if optimizer.lower() == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer.lower() == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Compile model
        self.model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\n[Step 2.2] Model Summary:")
        self.model.summary()
        
        return self.model
    
    def train_model(self, X_train, y_train, X_val, y_val, 
                   epochs=100, batch_size=32, verbose=1):
        """
        Train the neural network model
        
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
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        verbose : int
            Verbosity mode
            
        Returns:
        --------
        keras.callbacks.History
            Training history
        """
        print("\n[Step 2.3] Training Model...")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )
        
        print("\n[Step 2.4] Training completed!")
        print(f"  - Final training accuracy: {self.history.history['accuracy'][-1]:.4f}")
        print(f"  - Final validation accuracy: {self.history.history['val_accuracy'][-1]:.4f}")
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
            
        Returns:
        --------
        np.ndarray
            Predicted class labels
        """
        predictions = self.model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
            
        Returns:
        --------
        np.ndarray
            Prediction probabilities
        """
        return self.model.predict(X, verbose=0)
    
    def save_model(self, filepath):
        """Save model to file"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")