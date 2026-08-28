import tensorflow as tf
from tensorflow.keras import layers, models

def build_avalanche_classifier(input_shape=(224, 224, 3)):
    """
    Constructs a Convolutional Neural Network (CNN) for early avalanche 
    detection using image inputs (remote sensing or ground-based photographs).
    """
    model = models.Sequential([
        # Input Layer
        layers.Input(shape=input_shape),
        
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Fully Connected Classification Head
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary output: Avalanche (1) vs. Safe/Normal (0)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

if __name__ == "__main__":
    # Initialize the model as specified in the software stack (TensorFlow/Keras)[span_0](start_span)[span_0](end_span)
    avalanche_model = build_avalanche_classifier()
    avalanche_model.summary()
