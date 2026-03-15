import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from src.augment_image import get_data_augmentation


def build_model(num_classes,
                input_shape=(224, 224, 3),
                fine_tune=False):

    # GPU memory growth (prevents CUDA OOM)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    # Pretrained backbone
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Freeze layers initially
    base_model.trainable = fine_tune

    if fine_tune:
        # Unfreeze only top layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False

    # Data augmentation
    augmentation = get_data_augmentation()

    inputs = tf.keras.Input(shape=input_shape)

    x = augmentation(inputs)
    x = preprocess_input(x)

    x = base_model(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    return model