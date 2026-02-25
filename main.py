from src.preprocessing import preprocess
import os
from pathlib import Path
from src.train import train_model
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Memory growth enabled")
    except RuntimeError as e:
        print(e)

def main():
    preprocess(os.path.join(Path.cwd(), 'data'))
    train_model(
        dataset_path="data",
        epochs=25,
        batch_size=8,
        fine_tune=False
    )

if __name__ == "__main__":
    main()