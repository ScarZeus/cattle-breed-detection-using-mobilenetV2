import os
import tensorflow as tf
import numpy as np


def load_model_and_classes(model_path, dataset_path="data"):
    model = tf.keras.models.load_model(model_path)

    # Read class folders only
    class_names = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

    return model, class_names


def preprocess_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)

    # Normalize for MobileNetV2
    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_single_image(model, image_path, class_names):

    img_array = preprocess_image(image_path)

    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction)

    return class_names[predicted_index], confidence, prediction[0]


def predict_folder(model_path, test_folder, dataset_path="data"):

    model, class_names = load_model_and_classes(model_path, dataset_path)

    print("\nClasses:", class_names, "\n")

    for file in os.listdir(test_folder):

        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            image_path = os.path.join(test_folder, file)

            breed, confidence, probs = predict_single_image(
                model, image_path, class_names
            )

            print(f"\nImage: {file}")
            print(f"Predicted Breed: {breed}")
            print(f"Confidence: {confidence*100:.2f}%")

            # Show top 3 predictions
            top3 = np.argsort(probs)[-3:][::-1]

            print("Top Predictions:")
            for i in top3:
                print(f"  {class_names[i]} → {probs[i]*100:.2f}%")