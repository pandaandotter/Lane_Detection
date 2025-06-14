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

def preprocess_image(image_path: str, processing_mode: str, target_size: int = 80):
    img_full = cv2.imread(image_path)
    if img_full is None or img_full.shape[0] < 320 or img_full.shape[1] < 320:
        raise ValueError("Image must be at least 320x320 pixels.")

    x_start = (img_full.shape[1] - target_size) // 2
    y_start = (img_full.shape[0] - target_size) // 2
    x_end, y_end = x_start + target_size, y_start + target_size
    img_cropped = img_full[y_start:y_end, x_start:x_end]
    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)

    if processing_mode != "NO_SOBEL":
        img_gray = sobel_edge_filter(img_gray)

    img_input = np.expand_dims(img_gray, axis=-1).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)  # (1, 80, 80, 1)
    return img_cropped, img_input

def display_prediction(img_cropped, prediction, processing_mode):
    pred_img = (prediction * 255).astype(np.uint8)
    pred_color = cv2.applyColorMap(pred_img, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB), 0.6,
                              cv2.cvtColor(pred_color, cv2.COLOR_BGR2RGB), 0.4, 0)

    plt.figure(figsize=(4, 4))
    plt.imshow(overlay)
    plt.title(f"Prediction: {processing_mode}")
    plt.axis("off")
    plt.show()

def main(processing_mode: str):
    print(f"Running prediction with mode: {processing_mode}")
    image_path = "11.jpg"

    # Preprocess image
    img_cropped, img_input = preprocess_image(image_path, processing_mode)

    # Load model
    model_path = Path(__file__).parent / f"models/mobilenetv1_80_U-Net_1-{processing_mode.lower()}.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = load_model(model_path)

    # Predict
    pred = model.predict(img_input)[0, :, :, 0]  # (80, 80)

    # Show result
    display_prediction(img_cropped, pred, processing_mode)
