import os
import json
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

target_size = (40, 40)

# helper functions
def sobel_edge_filter(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    edge = np.clip(mag, 0, 255).astype(np.uint8)
    return edge
def max_pool_2x2(image):
    h, w = image.shape
    pooled = np.zeros((h // 2, w // 2), dtype=image.dtype)
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            block = image[i:i + 2, j:j + 2]
            pooled[i // 2, j // 2] = np.max(block)
    return pooled

def downsample_2x2_average(image):
    return cv2.resize(image, (40, 40), interpolation=cv2.INTER_AREA)

def downsample_2x2_sample(image):
    return image[::2, ::2]


def make_loader_fn(processing_mode: str, image_p:str, label_p: str):
    def load_image_and_mask_numpy_40(file_path):
        file_path = file_path.numpy().decode()
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        image_path = os.path.join(image_p, f"{file_name}.jpg")
        label_path = os.path.join(label_p, f"{file_name}.json")

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Crop center 80×80
        x_start, y_start = 120, 120
        x_end, y_end = 200, 200
        img_cropped = img[y_start:y_end, x_start:x_end]
        gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)

        if processing_mode == "NO_SOBEL":
            gray = downsample_2x2_average(gray)

        elif processing_mode == "SOBEL_BLURRED":
            gray = sobel_edge_filter(gray)
            gray = downsample_2x2_average(gray)

        elif processing_mode == "SOBEL_SAMPLED":
            gray = sobel_edge_filter(gray)
            gray = downsample_2x2_sample(gray)

        elif processing_mode == "SOBEL_MAX":
            gray = sobel_edge_filter(gray)
            gray = max_pool_2x2(gray)

        else:
            raise ValueError(f"Unknown processing mode: {processing_mode}")

        img_gray = np.expand_dims(gray, axis=-1).astype(np.float32) / 255.0

        # Load and downscale mask to match
        with open(label_path, 'r') as f:
            label = json.load(f)

        full_mask = np.zeros((80, 80), dtype=np.float32)
        for lane in label["lanes"]:
            coords = []
            for x, y in zip(lane, label["h_samples"]):
                if x >= 0 and 120 <= x < 200 and 120 <= y < 200:
                    x_local = x - 120
                    y_local = y - 120
                    coords.append((x_local, y_local))
            if len(coords) > 1:
                cv2.polylines(full_mask, [np.array(coords, dtype=np.int32)],
                              isClosed=False, color=(1.0,), thickness=1)

        mask_resized = max_pool_2x2(full_mask)
        """
        if processing_mode in {"NO_SOBEL", "SOBEL_BLURRED"}:
            mask_resized = cv2.resize(full_mask, (40, 40), interpolation=cv2.INTER_AREA)
        elif processing_mode == "SOBEL_SAMPLED":
            mask_resized = full_mask[::2, ::2]
        elif processing_mode == "SOBEL_MAX":
            mask_resized = max_pool_2x2(full_mask)
        else:
            raise ValueError(f"Unknown processing mode: {processing_mode}")
        """
        mask = np.expand_dims(mask_resized, axis=-1).astype(np.float32)



        return img_gray, mask

    return load_image_and_mask_numpy_40


def load_img_and_mask_tf(file_path, loader_fn):
    image, mask = tf.py_function(func=loader_fn,
                                 inp=[file_path],
                                 Tout=[tf.float32, tf.float32])
    image.set_shape((target_size[1], target_size[0], 1))
    mask.set_shape((target_size[1], target_size[0], 1))
    return image, mask


def create_dataset40(processing_mode: str, batch_size=8, DB_location:str = None):
    if DB_location == None:
        #model file
        overall_path = Path(__file__).parent.resolve()
        path = overall_path.parent.parent
        path.mkdir(exist_ok=True)

        image_dir = path/"Datasets"/ "TUSimple" / "resized_images"
        label_dir = path/"Datasets"/ "TUSimple" / "resized_labels"
    else:
        path = Path(DB_location)
        path.mkdir(exist_ok=True)

        image_dir = path / "resized_images"
        label_dir = path  / "resized_labels"

    loader_fn = make_loader_fn(processing_mode, image_dir.__str__(), label_dir.__str__())
    img_files = sorted([
        os.path.join(image_dir, fname)
        for fname in os.listdir(image_dir)
        if fname.endswith('.jpg')
    ])

    dataset = tf.data.Dataset.from_tensor_slices(img_files)
    dataset = dataset.map(lambda fp: load_img_and_mask_tf(fp, loader_fn), num_parallel_calls=tf.data.AUTOTUNE)
    if DB_location == None:
        # Create and use "cache/" folder
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        cache_path = cache_dir / f"train_cache_40_U-Net_{processing_mode.lower()}.cache"

        print(f"Caching dataset to: {cache_path.resolve()}")
        dataset = dataset.cache(str(cache_path))

        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    else:
        cache_path = "None"

    return dataset, cache_path

# ---------- Entry Point ----------

def main(processing_mode: str, DB_location:str = None):
    dataset, cache_path = create_dataset40(processing_mode, DB_location=DB_location)

    # sometimes the dataset cashing doesn't work and this becomes necessary
    count = 0
    for img_batch, mask_batch in dataset:
        count += 1

    print(f"Finished caching. Total batches processed: {count}")
    print(f"Grayscale 40×40 dataset loaded with processing mode: {processing_mode}")
    print(f"Cache written to: {cache_path.resolve()}")
    return dataset
