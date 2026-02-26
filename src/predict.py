import os
import tensorflow as tf
import numpy as np


def load_model_and_classes(model_path, dataset_path="data"):
    model = tf.keras.models.load_model(model_path)
    class_names = sorted(os.listdir(dataset_path))
    return model, class_names


def predict_single_image(model, image_path, class_names):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    if len(class_names) == 2:
        probability = float(prediction[0][0])
        predicted_index = int(probability > 0.5)
        confidence = probability if predicted_index == 1 else 1 - probability
    else:
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction)

    return class_names[predicted_index], confidence


def predict_folder(model_path, test_folder, dataset_path="data"):
    model, class_names = load_model_and_classes(model_path, dataset_path)

    print(f"\nClasses: {class_names}\n")

    for file in os.listdir(test_folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(test_folder, file)
            breed, confidence = predict_single_image(
                model, image_path, class_names
            )

            print(f"{file} → {breed} ({confidence*100:.2f}%)")