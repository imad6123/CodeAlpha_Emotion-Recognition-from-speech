import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ------------------------------------------------------------------
# STEP 1: Feature Extraction Pipeline (MFCCs)
# ------------------------------------------------------------------
def extract_features(file_path, max_pad_len=174):
    """
    Extracts MFCC features from an audio file and pads/truncates 
    the result to a uniform frame length.
    """
    # Load audio file (sampling rate = 22.05 kHz)
    audio, sr = librosa.load(file_path, res_type='kaiser_fast', sr=22050)
    
    # Extract 40 MFCC coefficients
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    
    # Pad or truncate to fixed length along the time axis
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
        
    return mfccs  # Shape: (40, 174)

# ------------------------------------------------------------------
# STEP 2: Model Architecture (1D-CNN + LSTM)
# ------------------------------------------------------------------
def build_speech_emotion_model(input_shape=(174, 40), num_classes=6):
    """
    Builds a CRNN model combining 1D Conv layers for local spectral patterns
    and an LSTM layer for temporal context.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Conv Block 1
    x = layers.Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Conv Block 2
    x = layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Recurrent Block (LSTM)
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    
    # Fully Connected Output Block
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Speech_Emotion_Classifier")
    return model

# ------------------------------------------------------------------
# STEP 3: Model Compilation
# ------------------------------------------------------------------
model = build_speech_emotion_model(input_shape=(174, 40), num_classes=6)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()