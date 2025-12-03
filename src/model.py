import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import callbacks


class NeuralNetworkModel:
    def __init__(self, input_dim, num_classes=3):
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
        
        inputs = tf.keras.Input(shape=(self.input_dim,))
        x = inputs
        
        for i, units in enumerate(layers_config):
            x = layers.Dense(units, activation=activation, 
                           name=f'hidden_{i+1}')(x)
            x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
            x = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax', 
                             name='output')(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='TelstraNN')
        
        if optimizer.lower() == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer.lower() == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
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
        print("\n[Step 2.3] Training Model...")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")
        
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
        predictions = self.model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        return self.model.predict(X, verbose=0)
    
    def save_model(self, filepath):
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")