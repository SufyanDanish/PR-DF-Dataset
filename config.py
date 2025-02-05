import os

# Paths to the datasets
TRAIN_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'training')
VAL_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'validation')
TEST_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'testing')

# Model and training parameters
NUM_EPOCHS = 50
BATCH_SIZE = 32
INPUT_SIZE = 224
NUM_CLASSES = 1

# Paths to save results
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, 'model', 'fire_detection_model.h5')
HISTORY_SAVE_PATH = os.path.join(RESULTS_DIR, 'metrics', 'training_history.json')
CONFUSION_MATRIX_SAVE_PATH = os.path.join(RESULTS_DIR, 'plots', 'confusion_matrix.png')
ROC_CURVE_SAVE_PATH = os.path.join(RESULTS_DIR, 'plots', 'roc_curve.png')
PRECISION_RECALL_CURVE_SAVE_PATH = os.path.join(RESULTS_DIR, 'plots', 'precision_recall_curve.png')
