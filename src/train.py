# train.py

import os
import tensorflow as tf
from src.model_builder import build_model
from src.load_data import load_datasets


def train_model(dataset_path="data",
                model_dir="checkpoints",
                epochs=25,
                batch_size=8,
                fine_tune=False):

    os.makedirs(model_dir, exist_ok=True)

    train_ds, val_ds, class_names = load_datasets(
        dataset_path,
        batch_size=batch_size
    )

    num_classes = len(class_names)

    # Resume training if checkpoint exists
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)

    if latest_checkpoint:
        print("Resuming from:", latest_checkpoint)
        model = tf.keras.models.load_model(latest_checkpoint)
    else:
        model = build_model(num_classes, fine_tune=fine_tune)

    # Learning rate adjustment
    lr = 1e-5 if fine_tune else 1e-4

    if num_classes == 2:
        loss_fn = "binary_crossentropy"
    else:
        loss_fn = "sparse_categorical_crossentropy"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=loss_fn,
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "ckpt_{epoch}.keras"),
        save_best_only=False
    )

    best_model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "best_model.keras"),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[
            checkpoint_callback,
            best_model_callback,
            early_stop,
            reduce_lr
        ]
    )

    model.save(os.path.join(model_dir, "final_model.keras"))

    return history