import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from pathlib import Path


def sobel_edge_filter(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return np.clip(mag, 0, 255).astype(np.uint8)

def sobel_max_pool_2x2(image):
    sobel = sobel_edge_filter(image)  # (80, 80)
    h, w = sobel.shape
    pooled = np.zeros((h // 2, w // 2), dtype=np.uint8)
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            block = sobel[i:i+2, j:j+2]
            pooled[i//2, j//2] = np.max(block)
    return pooled


def downsample_sample(img, target_size):
    factor = img.shape[0] // target_size
    return img[::factor, ::factor]


def preprocess_image(image_path: str, mode: str, target_size: int = 40):
    img_full = cv2.imread(image_path)
    if img_full is None or img_full.shape[0] < 320 or img_full.shape[1] < 320:
        raise ValueError("Image must be at least 320x320 pixels.")

    #  Crop center 80×80
    x_start = (img_full.shape[1] - 80) // 2
    y_start = (img_full.shape[0] - 80) // 2
    img_cropped = img_full[y_start:y_start+80, x_start:x_start+80]
    gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)

    #  Apply filters based on mode
    if mode == "NO_SOBEL":
        img_gray = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)

    elif mode == "SOBEL_BLURRED":
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = sobel_edge_filter(blurred)
        img_gray = cv2.resize(edges, (target_size, target_size), interpolation=cv2.INTER_AREA)

    elif mode == "SOBEL_SAMPLED":
        edges = sobel_edge_filter(gray)
        img_gray = downsample_sample(edges, target_size)

    elif mode == "SOBEL_MAX":
        img_gray = sobel_max_pool_2x2(gray)

    else:
        raise ValueError(f"Unknown processing mode: {mode}")

    # Prepare for model input
    img_input = np.expand_dims(img_gray, axis=-1).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    return img_cropped, img_input


def display_prediction(preprocessed_input, prediction, mode):
    # preprocessed_input is (40, 40, 1)
    base_img = (preprocessed_input[:, :, 0] * 255).astype(np.uint8)

    pred_img = (prediction * 255).astype(np.uint8)
    pred_color = cv2.applyColorMap(pred_img, cv2.COLORMAP_JET)
    base_color = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(base_color, 0.6, pred_color, 0.4, 0)

    plt.figure(figsize=(2, 2))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {mode}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main(processing_mode: str):
    print(f"Running prediction with mode: {processing_mode}")
    # placeholder... I need to fix this TODO
    image_path = "9.jpg"

    # Preprocess image
    img_cropped, img_input = preprocess_image(image_path, processing_mode)

    # Load model
    model_name = f"mobilenetv1_40_LightSeg-{processing_mode.lower()}.h5"#f"mobilenetv1_40_U-Net_1-{processing_mode.lower()}.h5"
    model_path = Path(__file__).parent / "models" / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = load_model(model_path)

    # Predict
    pred = model.predict(img_input)[0, :, :, 0]

    # Show result
    display_prediction(img_input[0], pred, processing_mode)


