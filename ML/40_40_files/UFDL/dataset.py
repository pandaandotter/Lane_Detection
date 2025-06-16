import os
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf

# ------------------------------------------------------------
# Paths helpers
# ------------------------------------------------------------
def get_dataset_paths(db_location=None):
    """Return [images_root, labels_root] for the (resized) TuSimple dataset."""
    if db_location is None:
        base = Path(__file__).resolve()
        for _ in range(3):
            base = base.parent
        dataset_path = base # / "Datasets" / "TUSimple"
    else:
        dataset_path = Path(db_location).expanduser().resolve()

    resized_images_path = dataset_path / "resized_images"
    resized_labels_path = dataset_path / "resized_labels"
    return [resized_images_path, resized_labels_path]


def get_image_file_list(db_location=None) -> List[str]:
    return sorted([
        os.path.join(get_dataset_paths(db_location)[0], fname)
        for fname in os.listdir(get_dataset_paths(db_location)[0])
        if fname.lower().endswith('.jpg')
    ])

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
target_size = (40, 40)         # width, height of model input
NUM_LANES   = 2
NUM_POINTS  = 20
NUM_COLS    = 20

# ------------------------------------------------------------
# Image processing helpers
# ------------------------------------------------------------
def sobel_edge_filter(image: np.ndarray) -> np.ndarray:
    gx  = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy  = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return np.clip(mag, 0, 255).astype(np.uint8)

def downsample_2x2_average(img: np.ndarray) -> np.ndarray:
    """Average‑pool 2×2 blocks → ½ resolution."""
    return cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2),
                      interpolation=cv2.INTER_AREA)

def downsample_2x2_sample(img: np.ndarray) -> np.ndarray:
    """Nearest‑neighbour sample every second pixel."""
    return img[::2, ::2]

def max_pool_2x2(img: np.ndarray) -> np.ndarray:
    """Max‑pool 2×2 blocks → ½ resolution."""
    return np.maximum.reduce([
        img[0::2, 0::2], img[0::2, 1::2],
        img[1::2, 0::2], img[1::2, 1::2]
    ])

# ------------------------------------------------------------
# Core loader
# ------------------------------------------------------------
def load_label_and_image_pair(image_path: str, processing_mode: str,
                              db_location=None
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read image + JSON label → tensors for training (40×40)."""
    file_path = image_path.numpy().decode() if hasattr(image_path, 'numpy') else image_path
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # --------------------------------------------------------
    # 1️⃣ Read and centre‑crop 80×80 from 320×… frame
    # --------------------------------------------------------
    img80 = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img80 is None:
        raise FileNotFoundError(f"Image not found: {file_path}")
    img80 = img80[120:200, 120:200]          # (80,80)

    # Always create no‑Sobel branch (average pooled 40×40)
    img_no_sobel = downsample_2x2_average(img80.copy())

    # --------------------------------------------------------
    # 2️⃣ Build gray channel according to processing_mode
    # --------------------------------------------------------
    if processing_mode == "NO_SOBEL":
        gray = downsample_2x2_average(img80.copy())

    elif processing_mode == "SOBEL_BLURRED":
        gray = sobel_edge_filter(img80.copy())
        gray = downsample_2x2_average(gray)

    elif processing_mode == "SOBEL_SAMPLED":
        gray = sobel_edge_filter(img80.copy())
        gray = downsample_2x2_sample(gray)

    elif processing_mode == "SOBEL_MAX":
        gray = sobel_edge_filter(img80.copy())
        gray = max_pool_2x2(gray)

    elif processing_mode == "DUAL":
        gray = sobel_edge_filter(img80.copy())
        gray = downsample_2x2_average(gray)

    else:
        raise ValueError(f"Unknown processing mode: {processing_mode}")

    # --------------------------------------------------------
    # 3️⃣ Build tf‑ready tensor (H,W,C) and normalise to [0,1]
    # --------------------------------------------------------
    if processing_mode == "DUAL":
        img_tensor = np.stack([img_no_sobel, gray], axis=-1).astype(np.float32) / 255.0
    else:
        img_tensor = np.expand_dims(gray, axis=-1).astype(np.float32) / 255.0

    # --------------------------------------------------------
    # 4️⃣ Read label JSON → class & X‑col targets in 40‑px space
    # --------------------------------------------------------
    label_path = os.path.join(get_dataset_paths(db_location)[1], f"{file_name}.json")
    with open(label_path, 'r') as f:
        label = json.load(f)

    class_target = np.zeros((NUM_LANES, NUM_POINTS), dtype=np.float32)
    x_target     = np.full( (NUM_LANES, NUM_POINTS), -1, dtype=np.int32)

    h_interval = 40 // NUM_POINTS         # 2 px
    h_samples  = list(range(0, 40, h_interval))   # 0,2,…,38

    # Sort candidate lanes by mean X after down‑sampling (L→R)
    lane_entries = []
    for lane_idx, lane in enumerate(label["lanes"]):
        coords_80 = [(x - 120, y - 120)
                     for x, y in zip(lane, label["h_samples"])
                     if 120 <= x < 200 and 120 <= y < 200]
        #print(f"Image {file_name} – lane {lane_idx}: {len(coords_80)} points inside crop")

        coords_40 = [(xc // 2, yc // 2) for xc, yc in coords_80]

        if len(coords_40) >= 2:
            avg_x = np.mean([x for x, _ in coords_40])
            lane_entries.append((avg_x, coords_40))


    lane_entries = sorted(lane_entries, key=lambda t: t[0])[:NUM_LANES]

    for lane_idx, (_, coords) in enumerate(lane_entries):
        coords = sorted(coords, key=lambda pt: pt[1])   # by y
        xs, ys = zip(*coords)
        interp_xs = np.interp(h_samples, ys, xs, left=np.nan, right=np.nan)

        for i, (x, _) in enumerate(zip(interp_xs, h_samples)):
            if not np.isnan(x) and 0 <= x < 40:
                class_target[lane_idx, i] = 1.0
                x_target  [lane_idx, i] = int(x / 40.0 * NUM_COLS)  # 0‑19 bin
    #print("class_target[0].sum():", class_target[0].sum())
    #print("class_target[1].sum():", class_target[1].sum())
    return img_tensor, class_target.flatten(), x_target.flatten().astype(np.int32)

# ------------------------------------------------------------
# tf.data pipeline
# ------------------------------------------------------------
def create_dataset40(processing_mode: str,
                     batch_size=8,
                     DB_location=None):
    """Return a tf.data.Dataset yielding (image, targets) tuples."""
    def tf_wrapper(image_path):
        img, lane_exist, lane_pos = tf.py_function(
            func=lambda path: load_label_and_image_pair(path, processing_mode, DB_location),
            inp=[image_path],
            Tout=(tf.float32, tf.float32, tf.int32)
        )
        img.set_shape((40, 40, 2 if processing_mode == "DUAL" else 1))
        lane_exist.set_shape([NUM_LANES * NUM_POINTS])
        lane_pos  .set_shape([NUM_LANES * NUM_POINTS])
        return img, {"lane_pos": lane_pos, "lane_exist": lane_exist}

    img_files = get_image_file_list(DB_location)
    dataset = tf.data.Dataset.from_tensor_slices(img_files)
    dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    if batch_size:
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset, "fooooooooooool"

if __name__ == "__main__":  # pragma: no cover
    # quick smoke‑test
    ds = create_dataset40("DUAL", batch_size=1)
    for img, tgt in ds.take(1):
        print(img.shape, tgt['lane_exist'].shape)
