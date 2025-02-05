from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import TRAIN_PATH, VAL_PATH, TEST_PATH, BATCH_SIZE, INPUT_SIZE

def create_data_generators():
    print("Initializing data generators...")
    train_aug = ImageDataGenerator()
    val_aug = ImageDataGenerator()
    test_aug = ImageDataGenerator()

    train_gen = train_aug.flow_from_directory(TRAIN_PATH, target_size=(INPUT_SIZE, INPUT_SIZE), color_mode='rgb', batch_size=BATCH_SIZE, class_mode='binary', shuffle=True)
    val_gen = val_aug.flow_from_directory(VAL_PATH, target_size=(INPUT_SIZE, INPUT_SIZE), color_mode='rgb', batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)
    test_gen = test_aug.flow_from_directory(TEST_PATH, target_size=(INPUT_SIZE, INPUT_SIZE), color_mode='rgb', batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)

    print("Data generators initialized.")
    return train_gen, val_gen, test_gen
