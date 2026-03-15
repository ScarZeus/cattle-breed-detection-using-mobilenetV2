import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

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

    # Load trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load class names
    class_names = get_class_names(DATASET_PATH)

    print("\nDetected Classes:", class_names)

    # Load image
    img = tf.keras.utils.load_img(IMAGE_PATH, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)

    # Apply same preprocessing used during training
    img_array = preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction[0])
    confidence = np.max(prediction[0])

    print("\nImage:", IMAGE_PATH)
    print("Predicted Breed:", class_names[predicted_index])
    print(f"Confidence: {confidence * 100:.2f}%")

    # Optional: show probabilities for all breeds
    print("\nAll Breed Probabilities:")
    for i, breed in enumerate(class_names):
        print(f"{breed}: {prediction[0][i] * 100:.2f}%")


if __name__ == "__main__":
    main()