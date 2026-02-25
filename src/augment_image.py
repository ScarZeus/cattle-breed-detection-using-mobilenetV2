# augmentation.py

import tensorflow as tf
from tensorflow.keras import layers


def get_data_augmentation():
    """
    Returns a Sequential data augmentation pipeline
    suitable for cattle breed classification.
    """
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    return data_augmentation