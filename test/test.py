# test_single.py

import numpy as np
import tensorflow as tf
import os

MODEL_PATH = "checkpoints/best_model.keras"
IMAGE_PATH = "/home/k-kevin-gladson/git-applications/cattle-breed-detection-using-mobilenetV2/test/test_img/test2.jpeg"
DATASET_PATH = "data"
IMG_SIZE = (224, 224)


def get_class_names(dataset_path):
    return sorted([
        folder for folder in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, folder))
    ])


def main():
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Get class names
    class_names = get_class_names(DATASET_PATH)

    # Load image
    img = tf.keras.utils.load_img(IMAGE_PATH, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array, verbose=0)

    if len(class_names) == 2:
        probability = float(prediction[0][0])
        predicted_index = int(probability > 0.5)
        confidence = probability if predicted_index == 1 else 1 - probability
    else:
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction)

    print("\nImage:", IMAGE_PATH)
    print("Predicted Breed:", class_names[predicted_index])
    print(f"Confidence: {confidence * 100:.2f}%\n")


if __name__ == "__main__":
    main()