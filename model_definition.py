import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB2 as XModel
from config import INPUT_SIZE, NUM_CLASSES

def create_model():
    print("Building model...")
    inputs = tf.keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    base_model = XModel(weights='imagenet', input_shape=(INPUT_SIZE, INPUT_SIZE, 3), include_top=False)
    base_model_input = base_model(inputs)
    x10 = layers.GlobalAveragePooling2D()(base_model_input)
    x = layers.BatchNormalization()(base_model_input)

    # Pyramidial Attention
    x2 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(x)
    x2 = layers.GlobalAveragePooling2D()(x2)
    x2 = layers.BatchNormalization()(x2)

    x3 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(x)
    x3 = layers.GlobalAveragePooling2D()(x3)
    x3 = layers.BatchNormalization()(x3)

    x4 = layers.Conv2D(filters=64, kernel_size=(7, 7), activation='relu', padding='same')(x)
    x5 = layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(x4)
    x6 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x5)

    x7 = layers.Conv2D(filters=64, kernel_size=(7, 7), activation='relu', padding='same')(x4)
    x8 = layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(x5)
    x9 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x6)

    x8 = layers.GlobalAveragePooling2D()(x8)
    x8 = layers.BatchNormalization()(x8)
    x9 = layers.GlobalAveragePooling2D()(x9)
    x9 = layers.BatchNormalization()(x9)

    F1 = layers.concatenate([x8, x9])
    x7 = layers.GlobalAveragePooling2D()(x7)
    x7 = layers.BatchNormalization()(x7)
    F2 = layers.concatenate([F1, x7])
    F3 = layers.concatenate([F2, x2])
    F4 = layers.concatenate([F3, x3])
    PF = layers.BatchNormalization()(F4)

    PF = layers.Dropout(0.2)(PF)
    FF = layers.concatenate([x10, PF])
    F = layers.Dense(512, activation='relu')(FF)
    F = layers.Dense(512, activation='relu')(F)
    F = layers.BatchNormalization()(F)
    F = layers.Dense(512, activation='relu')(F)
    outputs = layers.Dense(units=NUM_CLASSES, activation='sigmoid')(F)

    model = tf.keras.Model(inputs, outputs)
    print("Model built.")
    return model
