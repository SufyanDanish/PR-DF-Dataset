import os
from training import train_model
from evaluation import evaluate_model
from config import INPUT_SIZE, RESULTS_DIR
from PIL import Image, ImageFile

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def check_images(directory):
    corrupted_images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Verify that it is, in fact, an image
                except (IOError, SyntaxError) as e:
                    print(f"Corrupted image: {file_path}")
                    corrupted_images.append(file_path)
    return corrupted_images

def remove_corrupted_images(corrupted_images):
    for image in corrupted_images:
        try:
            os.remove(image)
            print(f"Removed corrupted image: {image}")
        except Exception as e:
            print(f"Failed to remove corrupted image: {image}. Error: {e}")

if __name__ == "__main__":
    # Ensure the results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'model'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'metrics'), exist_ok=True)

    print("Checking for corrupted images...")
    train_corrupted = check_images('dataset/training')
    val_corrupted = check_images('dataset/validation')
    test_corrupted = check_images('dataset/testing')

    remove_corrupted_images(train_corrupted)
    remove_corrupted_images(val_corrupted)
    remove_corrupted_images(test_corrupted)
    print("Corrupted images removed.")

    print("Starting the training process...")
    try:
        model, history = train_model()
        print("Training completed.")
        evaluate_model(model)
    except Exception as e:
        print(f"An error occurred: {e}")
