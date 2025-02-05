import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from config import NUM_EPOCHS, MODEL_SAVE_PATH, HISTORY_SAVE_PATH
from data_handling import create_data_generators
from model_definition import create_model
import json

def train_model():
    print("Creating data generators...")
    train_gen, val_gen, _ = create_data_generators()
    print("Data generators created.")

    print("Creating model...")
    model = create_model()
    print("Model created.")

    opt = Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
    model.summary()

    print("Starting model training...")
    history = model.fit(train_gen, validation_data=val_gen, epochs=NUM_EPOCHS)
    print("Model training completed.")

    # Save the model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # Save the training history
    with open(HISTORY_SAVE_PATH, 'w') as f:
        json.dump(history.history, f)
    print(f"Training history saved to {HISTORY_SAVE_PATH}")

    return model, history
