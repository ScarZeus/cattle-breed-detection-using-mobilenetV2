import os
from pathlib import Path

def preprocess(path):
    sub_dirs = {
        "red_angus": "red_agnus",          
        "belted_galloway": "belted_galloway"
    }
    rename_files(path, sub_dirs)


def rename_files(path, sub_dirs):
    for label, folder in sub_dirs.items():
        dir_path = os.path.join(path, folder)

        if not os.path.isdir(dir_path):
            print(f"Skipping missing folder: {dir_path}")
            continue

        print(f"Processing: {dir_path}")

        files = sorted(os.listdir(dir_path))

        index = 0
        for filename in files:
            old_path = os.path.join(dir_path, filename)

            if not os.path.isfile(old_path):
                continue

            if filename.startswith(label + "_"):
                continue

            file_ext = os.path.splitext(filename)[1]
            new_name = f"{label}_{index}{file_ext}"
            new_path = os.path.join(dir_path, new_name)

            if os.path.exists(new_path):
                print(f"Skipping existing file: {new_name}")
                continue

            os.rename(old_path, new_path)
            print(f"Renamed: {filename} → {new_name}")

            index += 1


preprocess(os.path.join(Path.cwd(), 'data'))