# model_builder.py

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from src.augment_image import get_data_augmentation


def build_model(num_classes,
                input_shape=(224, 224, 3),
                fine_tune=False):

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    base_model.trainable = fine_tune

    if fine_tune:
        # Unfreeze only last 20 layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False

    augmentation = get_data_augmentation()

    inputs = tf.keras.Input(shape=input_shape)
    x = augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    return model