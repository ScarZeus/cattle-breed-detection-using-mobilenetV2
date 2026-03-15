import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
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

    print("Detected Classes:", class_names)
    print("Total Classes:", num_classes)

    latest_checkpoint = tf.train.latest_checkpoint(model_dir)

    if latest_checkpoint:
        print("Resuming from:", latest_checkpoint)
        model = tf.keras.models.load_model(latest_checkpoint)
    else:
        model = build_model(num_classes, fine_tune=fine_tune)

    lr = 1e-5 if fine_tune else 1e-4

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
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

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint_callback, best_model_callback]
    )

    # -------- Evaluate metrics on validation set --------
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        preds = np.argmax(preds, axis=1)

        y_pred.extend(preds)
        y_true.extend(labels.numpy())

    print("\n===== Classification Report =====")
    print(classification_report(y_true, y_pred, target_names=class_names))

    model.save(os.path.join(model_dir, "final_model.keras"))

    return history