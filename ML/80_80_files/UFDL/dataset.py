import os
import json
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple

def get_dataset_paths(db_location=None):
    if db_location is None:
        # Start from the current file's location
        base_path = Path(__file__).resolve().parent.parent.parent.parent
        dataset_path = base_path / 'Datasets' / 'TUSimple'
    else:
        dataset_path = Path(db_location).resolve()

    resized_images_path = dataset_path / 'resized_images'
    resized_labels_path = dataset_path / 'resized_labels'

    return [resized_images_path, resized_labels_path]



#image_dir = "resized_images"
#label_dir = "resized_labels"
target_size = (80, 80)
NUM_LANES = 2
NUM_POINTS = 20
NUM_COLS = 20

def sobel_edge_filter(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    edge = np.clip(mag, 0, 255).astype(np.uint8)
    return edge

def get_image_file_list(db_location=None) -> List[str]:
    image_files = sorted([
        os.path.join(get_dataset_paths(db_location)[0], fname)
        for fname in os.listdir(get_dataset_paths(db_location)[0])
        if fname.lower().endswith('.jpg')
    ])
    valid_image_files = []
    for img_path in image_files:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(get_dataset_paths(db_location)[1], f"{base}.json")
        if os.path.isfile(label_path):
            valid_image_files.append(img_path)

    print(f"Found {len(valid_image_files)} images with labels")
    return valid_image_files

def load_label_and_image_pair(image_path: str, processing_mode: str, db_location=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    file_path = image_path.numpy().decode() if hasattr(image_path, 'numpy') else image_path
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {file_path}")

    # Crop 80x80 center
    img = img[120:200, 120:200]

    img_no_sobel = img.copy()

    # Preprocess
    if processing_mode == "NO_SOBEL":
        img_proc = img.copy()  # identical, just repeat
    elif processing_mode in ["SOBEL_BLURRED","SOBEL_BLURRED","SOBEL_SAMPLED","SOBEL_MAX", "DUAL"]:
        img_proc = sobel_edge_filter(img)
    else:
        raise ValueError(f"Unknown processing mode: {processing_mode}")



    if processing_mode == "DUAL":
        img_tensor = np.stack([img_no_sobel, img_proc], axis=-1).astype(np.float32) / 255.0
    else:
        img_tensor = np.expand_dims(img, axis=-1).astype(np.float32) / 255.0

    # Load label
    label_path = os.path.join(get_dataset_paths(db_location)[1], f"{file_name}.json")
    with open(label_path, 'r') as f:
        label = json.load(f)

    class_target = np.zeros((NUM_LANES, NUM_POINTS), dtype=np.float32)
    x_target = np.full((NUM_LANES, NUM_POINTS), fill_value=-1, dtype=np.int32)

    h_interval = 80 // NUM_POINTS
    h_samples = list(range(0, 80, h_interval))

    # Sort lanes by average x (left to right)
    lane_entries = []
    for lane in label["lanes"]:
        valid_coords = [(x - 120, y - 120) for x, y in zip(lane, label["h_samples"])
                        if 120 <= x < 200  and 120 <= y < 200]
        if len(valid_coords) >= 2:
            avg_x = np.mean([x for x, _ in valid_coords])
            lane_entries.append((avg_x, valid_coords))

    lane_entries = sorted(lane_entries, key=lambda x: x[0])[:NUM_LANES]

    for lane_idx, (_, valid_coords) in enumerate(lane_entries):
        valid_coords = sorted(valid_coords, key=lambda pt: pt[1])
        xs, ys = zip(*valid_coords)
        interp_xs = np.interp(h_samples, ys, xs, left=np.nan, right=np.nan)

        for i, (x, y) in enumerate(zip(interp_xs, h_samples)):
            if not np.isnan(x) and 0 <= x < 80:
                class_target[lane_idx, i] = 1.0
                x_target[lane_idx, i] = int(x / 80.0 * NUM_COLS)  # normalize to 80px image width


    return img_tensor, class_target.flatten(), x_target.flatten().astype(np.int32)

def create_dataset40(processing_mode: str, batch_size=8,DB_location=None):

    def tf_wrapper(image_path):
        img, lane_exist, lane_pos = tf.py_function(
            func=lambda path: load_label_and_image_pair(path, processing_mode),
            inp=[image_path],
            Tout=(tf.float32, tf.float32, tf.int32)
        )
        if processing_mode == "DUAL":
            img.set_shape((80,80,2))
        else:
            img.set_shape((80, 80, 1))
        lane_exist.set_shape([NUM_LANES * NUM_POINTS])
        lane_pos.set_shape([NUM_LANES * NUM_POINTS])
        return img, {
            'lane_pos': lane_pos,
            'lane_exist': lane_exist
        }

    img_files = get_image_file_list(DB_location)
    dataset = tf.data.Dataset.from_tensor_slices(img_files)
    dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)


    return dataset

def main(processing_mode: str, DB_location:str =None):
    return create_dataset40(processing_mode, DB_location = DB_location)




