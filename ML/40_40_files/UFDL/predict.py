
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from pathlib import Path
import collections

def sobel_edge_filter(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return np.clip(mag, 0, 255).astype(np.uint8)

def sobel_max_pool_2x2(image):
    sobel = sobel_edge_filter(image)
    h, w = sobel.shape
    pooled = np.zeros((h // 2, w // 2), dtype=np.uint8)
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            block = sobel[i:i + 2, j:j + 2]
            pooled[i // 2, j // 2] = np.max(block)
    return pooled

def downsample_sample(img, target_size):
    factor = img.shape[0] // target_size
    return img[::factor, ::factor]

def preprocess_image(image_path: str, mode: str, target_size: int = 40):
    img_full = cv2.imread(image_path)
    if img_full is None or img_full.shape[0] < 320 or img_full.shape[1] < 320:
        raise ValueError("Image must be at least 320x320 pixels.")

    x_start = (img_full.shape[1] - 80) // 2
    y_start = (img_full.shape[0] - 80) // 2
    img_cropped = img_full[y_start:y_start + 80, x_start:x_start + 80]
    gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)

    img_gray = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
    res = img_gray.copy()
    if mode == "NO_SOBEL":
        img_proc = img_gray.copy()  # identical, just repeat
    elif mode in ["SOBEL_BLURRED", "SOBEL_BLURRED", "SOBEL_SAMPLED", "SOBEL_MAX", "DUAL"]:
        img_proc = sobel_edge_filter(img_gray)
    else:
        raise ValueError(f"Unknown processing mode: {mode}")

    if mode == "DUAL":
        img_input =  np.stack([res, img_proc], axis=-1).astype(np.float32) / 255.0
        img_input = np.expand_dims(img_input, axis=0)
    else:
        img_input = np.expand_dims(img_proc, axis=-1).astype(np.float32) / 255.0
        img_input = np.expand_dims(img_input, axis=0)
    return img_cropped, img_input

def display_prediction(cropped_img, presence, col_bins,
                       mode, num_cols=20, num_lanes=2, num_points=20):
    img = cropped_img.copy()
    step = 80 // num_points              # vertical distance between rows
    """
    for flat_idx in range(num_lanes * num_points):
        if presence[flat_idx] <= 0.5:
            continue                     # skip “no lane” rows
        lane_id    = flat_idx // num_points          # 0 or 1
        point_id   = flat_idx %  num_points          # 0 … 19
        y = point_id * step + step // 2              # same y for every lane
        x_bin      = int(col_bins[flat_idx])
        x = int((x_bin + 0.5) * (80 / num_cols))

        color = (0, 255, 0) if lane_id == 0 else (0, 255, 255)
        cv2.circle(img, (x, y), 2, color, -1)
    """
    NUM_LANES = 2
    NUM_POINTS = 20
    NUM_COLS = 20

    for lane in range(NUM_LANES):
        for i in range(NUM_POINTS):
            flat_idx = lane * NUM_POINTS + i

            if presence[flat_idx] <= 0.5:
                continue

            x_bin = col_bins[flat_idx]
            x = int((x_bin + 0.5) * (80 / NUM_COLS))  # map bin back to 80 px crop
            y = int(i * (80 / NUM_POINTS))  # y by point index

            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {mode}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main(processing_mode: str):
    print(f"Running prediction with mode: {processing_mode}")
    image_path = "11.jpg"  # Replace with your test image path

    img_cropped, img_input = preprocess_image(image_path, processing_mode)
    model_name = f"UFDL_2_{processing_mode.lower()}.h5"
    model_path = Path(__file__).parent / "models" / model_name
    print(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load_model(model_path, compile=False)

    outputs = model.predict(img_input)
    if isinstance(outputs, list):
        lane_pos = outputs[0]
        lane_exist = outputs[1]
    elif isinstance(outputs, dict):
        lane_pos = outputs["lane_pos"]
        lane_exist = outputs["lane_exist"]
    else:
        raise TypeError(f"Unexpected model output type: {type(outputs)}")

    # Get the most likely bin index per row
    presence = np.array(lane_exist[0]).flatten()
    col_bins = np.argmax(lane_pos[0], axis=-1)

    counter = collections.Counter(np.atleast_1d(col_bins))
    print("presence.shape =", presence.shape)  # should be (40,)
    print("col_bins.shape =", col_bins.shape)
    display_prediction(img_cropped, presence, col_bins, processing_mode)
