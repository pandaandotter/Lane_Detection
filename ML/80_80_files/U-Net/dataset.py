import os
import json
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple

image_dir = "resized_images"
label_dir = "resized_labels"
target_size = (80, 80)
NUM_LANES = 2
NUM_POINTS = 20

def sobel_edge_filter(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    edge = np.clip(mag, 0, 255).astype(np.uint8)
    return edge

def get_image_file_list() -> List[str]:
    return sorted([
        os.path.join(image_dir, fname)
        for fname in os.listdir(image_dir)
        if fname.endswith(".jpg")
    ])

def load_label_and_image_pair(image_path: str, processing_mode: str) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    file_path = image_path.numpy().decode() if hasattr(image_path, 'numpy') else image_path
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    label_path = os.path.join(label_dir, f"{file_name}.json")

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {file_path}")

    # Crop 80x80 center
    img = img[120:200, 120:200]

    if processing_mode in {"SOBEL_BLURRED","SOBEL_SAMPLED","SOBEL_MAX"}:
        img = sobel_edge_filter(img)

    img_tensor = np.expand_dims(img, axis=-1).astype(np.float32) / 255.0
    with open(label_path, 'r') as f:
        label = json.load(f)

    mask = np.zeros((target_size[1], target_size[0], 1), dtype=np.float32)

    for lane in label["lanes"]:
        coords = []
        for x, y in zip(lane, label["h_samples"]):
            if x >= 0 and 120 <= x < 200 and 120 <= y < 200:
                x_local = x - 120
                y_local = y - 120
                coords.append((x_local, y_local))
        if len(coords) > 1:
            cv2.polylines(mask, [np.array(coords, dtype=np.int32)],
                          isClosed=False, color=(1.0,), thickness=1)
    return img_tensor, mask.astype(np.float32)

def create_dataset80(processing_mode: str, batch_size=8):
    def tf_wrapper(image_path):
        img, mask = tf.py_function(
            func=lambda path: load_label_and_image_pair(path, processing_mode),
            inp=[image_path],
            Tout=(tf.float32, tf.float32)
        )
        img.set_shape((target_size[1], target_size[0], 1))
        mask.set_shape((target_size[1], target_size[0], 1))
        return img, mask

    img_files = get_image_file_list()
    dataset = tf.data.Dataset.from_tensor_slices(img_files)
    dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

    # Cache to disk
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"train_cache_80_seg_{processing_mode.lower()}.cache"
    print(f"Caching dataset to: {cache_path.resolve()}")

    dataset = dataset.cache(str(cache_path))
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset, cache_path

def main(processing_mode: str):
    dataset, cache_path = create_dataset80(processing_mode)
    count = 0
    for img_batch, label_batch in dataset:
        count += 1
    print(f"Finished caching. Total batches: {count}")
    print(f"Dataset ready with mode: {processing_mode}")
    print(f"Cache path: {cache_path.resolve()}")
