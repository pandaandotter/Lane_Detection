import numpy as np
import cv2
import tensorflow as tf


def extract_points_unet(pred_heatmap, threshold=0.85):
    coords = np.argwhere(pred_heatmap >= threshold)  # Returns (y, x) format
    return [(x, y) for y, x in coords]
def lane_distance_error_unet(pred_heatmap, label, size):
    pred_points = extract_points_unet(pred_heatmap)

    # Get ground truth mask from label
    label_points =extract_points_unet(label[:,:,0])

    if len(pred_points) < 2 or len(label_points) < 2:
        return None  # Not enough data to fit

    x_vals = np.linspace(0, size - 1, num=size)

    pred_poly1 = fit_lane_function(pred_points, degree=1)
    pred_poly2 = fit_lane_function(pred_points, degree=2)
    label_poly1 = fit_lane_function(label_points, degree=1)
    label_poly2 = fit_lane_function(label_points, degree=2)

    if any(p is None for p in (pred_poly1, pred_poly2, label_poly1, label_poly2)):
        return None

    y_pred_avg = (np.polyval(pred_poly1, x_vals) + np.polyval(pred_poly2, x_vals)) / 2
    y_label_avg = (np.polyval(label_poly1, x_vals) + np.polyval(label_poly2, x_vals)) / 2

    error = np.abs(y_pred_avg - y_label_avg).mean()
    return error

def compute_fp_fn_tp_rates_unet(pred_heatmap, label, size, threshold= 0.85):
    true_mask = label[:,:,0]
    pred_binary = (pred_heatmap >= threshold).astype(np.uint8)

    # TP: predicted 1, label 1
    true_positives = np.logical_and(pred_binary == 1, true_mask == 1).sum()
    # FP: predicted 1, label 0
    false_positives = np.logical_and(pred_binary == 1, true_mask == 0).sum()
    # FN: predicted 0, label 1
    false_negatives = np.logical_and(pred_binary == 0, true_mask == 1).sum()
    true_negatives = tf.reduce_sum((1.0 - true_mask) * (1.0 - pred_binary))

    total_predicted = tf.reduce_sum(tf.cast(pred_binary, tf.float32))
    total_actual = tf.reduce_sum(tf.cast(true_mask, tf.float32))

    fp_rate = false_positives / (total_predicted + 1e-6)  # avoid division by zero
    fn_rate = false_negatives / (total_actual + 1e-6)
    tp_rate = true_positives / (total_actual + 1e-6)

    total = false_positives + false_negatives + true_negatives + true_positives
    accuracy = (true_positives + true_negatives) / (total + 1e-6)

    return fp_rate, fn_rate, tp_rate, float(accuracy)


def compute_fp_fn_tp_rates_ufdl(prediction, label_dict, size):
    presence, col_bins = prediction

    true_presence = label_dict['lane_exist']
    threshold = 0.5

    pred_binary = tf.cast(presence >= threshold, tf.float32)
    true_mask = tf.cast(true_presence >= threshold, tf.float32)

    TP = tf.reduce_sum(true_mask * pred_binary)
    FP = tf.reduce_sum((1.0 - true_mask) * pred_binary)
    FN = tf.reduce_sum(true_mask * (1.0 - pred_binary))
    TN = tf.reduce_sum((1.0 - true_mask) * (1.0 - pred_binary))
    total_pred = tf.reduce_sum(pred_binary)
    total_actual = tf.reduce_sum(true_mask)

    # Avoid division by zero
    tp_rate = TP / (total_actual + 1e-6)
    fp_rate = FP / (total_pred + 1e-6)
    fn_rate = FN / (total_actual + 1e-6)

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / (total + 1e-6)

    return float(fp_rate), float(fn_rate), float(tp_rate), float(accuracy)



def fit_lane_function(points, degree=1):
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    if len(x) < degree + 1:
        return None  # not enough points
    return np.polyfit(x, y, degree)


def lane_distance_error(pred, label, size = 80):
    pred_presence, pred_pos = pred
    label_presence, label_pos = label

    # Extract (x, y) points
    pred_points = extract_points_ufdl(pred_presence, pred_pos, image_width=size)
    label_points = extract_points_ufdl(label_presence, label_pos, image_width=size)

    if not pred_points or not label_points:
        return None  # not enough data to compare

    x_vals = np.linspace(0, 39, num=40)

    pred_poly1 = fit_lane_function(pred_points, degree=1)
    pred_poly2 = fit_lane_function(pred_points, degree=2)
    label_poly1 = fit_lane_function(label_points, degree=1)
    label_poly2 = fit_lane_function(label_points, degree=2)

    if any(p is None for p in (pred_poly1, pred_poly2, label_poly1, label_poly2)):
        return None

    y_pred_avg = (np.polyval(pred_poly1, x_vals) + np.polyval(pred_poly2, x_vals)) / 2
    y_label_avg = (np.polyval(label_poly1, x_vals) + np.polyval(label_poly2, x_vals)) / 2

    error = np.abs(y_pred_avg - y_label_avg).mean()
    return error


def extract_points_ufdl(lane_exist, lane_pos, threshold=0.5, num_cols=20, image_width=80):
    points = []
    col_width = image_width / num_cols  # 80 / 20 = 4.0

    lane_exist = tf.convert_to_tensor(lane_exist)
    lane_pos = tf.convert_to_tensor(lane_pos)

    lane_exist = lane_exist.numpy()
    lane_pos = lane_pos.numpy()

    for y, (exist, bin_idx) in enumerate(zip(lane_exist, lane_pos)):
        if exist >= threshold and bin_idx >= 0:
            x = float(bin_idx) * col_width + col_width / 2  # Center of bin
            points.append((x, y))

    return points



