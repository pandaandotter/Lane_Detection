import tracemalloc
from timeit import default_timer as timer
from warnings import catch_warnings

from Run_Test import compute_fp_fn_tp_rates_unet, compute_fp_fn_tp_rates_ufdl, lane_distance_error, lane_distance_error_unet
import numpy as np
from tensorflow.keras.models import load_model
from ml_run_test import run_task
import cv2
import tensorflow as tf
import csv


def train(modelName, type, model_size, train_set, val_set):
    return run_task("train", model_size, modelName, type, train_set=train_set,val_set=val_set)

def get_training_set(modelName, type, model_size, dataset_path):
    dataset = run_task("dataset", model_size, modelName, type, dataset_path=dataset_path)
    dataset_size = len(dataset)#cardinality().numpy()
    train_size = int(dataset_size * 0.8)
    other_size = int(dataset_size * 0.1)
    dataset = dataset.cache()
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size).take(other_size)
    test_ds = dataset.skip(train_size + other_size)

    return train_ds, val_ds, test_ds



def predict(model_test, modelName, img):
    # allegedly model(img) is an option and recommended TODO: investigate
    if modelName == "U-Net":
        if isinstance(img, tf.Tensor):
            img = img.numpy()
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Normalize and expand dims
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)  # (H, W, 1)
        img_tensor = np.expand_dims(img, axis=0)  # (1, H, W, 1)

        tracemalloc.clear_traces()
        tracemalloc.start()

        prediction = model_test.predict(img_tensor)[0, :, :, 0]
        current, peak = tracemalloc.get_traced_memory()

        start = timer()
        dummy_predict = model_test.predict(img_tensor)
        end = timer()

        return prediction, (current, peak), (end - start) * 1000

    if modelName == "UFDL":
        if isinstance(img, tf.Tensor):
            img = img.numpy()

        # If RGB, convert to grayscale
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Normalize and expand dims
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)  # (80, 80, 1)
        img_tensor = np.expand_dims(img, axis=0)  # (1, 80, 80, 1)
        tracemalloc.clear_traces()
        tracemalloc.start()  # ram_measure_start()
        outputs = model_test.predict(img)
        current, peak = tracemalloc.get_traced_memory()
        start = timer()
        dummmy_predict_for_time = model_test.predict(img_tensor)
        end = timer()

        if isinstance(outputs, list):
            lane_pos = outputs[0]
            lane_exist = outputs[1]
        elif isinstance(outputs, dict):
            lane_pos = outputs["lane_pos"]
            lane_exist = outputs["lane_exist"]
        else:
            raise TypeError(f"Unexpected model output type: {type(outputs)}")
        # get the most likely bin index per row
        presence = np.array(lane_exist[0]).flatten()
        col_bins = np.argmax(lane_pos[0], axis=-1)

        return (presence, col_bins), (current , peak), (end - start) * 1000

def accuracy(prediciton, label, size, modelName):
    if modelName == "U-Net":
        return compute_fp_fn_tp_rates_unet(prediciton, label, size)
    if modelName == "UFDL":
        return compute_fp_fn_tp_rates_ufdl(prediciton, label, size)

def curve_accuracy(prediction, label_dict, size):
    if modelName == "UFDL":
        pred_presence, pred_bins = prediction[0], prediction[1]
        true_presence = label_dict["lane_exist"]
        true_positions = label_dict["lane_pos"]
        return lane_distance_error((pred_presence, pred_bins), (true_presence, true_positions), size=size)
    if modelName == "U-Net":
        return lane_distance_error_unet(prediction, label_dict, size)





taskS = ["dataset", "predict", "train", "view_label"]
sizeS = {40: "40_40_files", 80: "80_80_files"}  # in pixels
modelS = ["U-Net", "UFDL"]  # CNN not implemented yet
paramsS = ["NO_SOBEL", "SOBEL_BLURRED", "SOBEL_SAMPLED", "SOBEL_MAX", "DUAL"]


models = ["UFDL"]
preprocessing_type = ["NO_SOBEL"]
for i in range(10):

    for modelName in models:
        for size in [40, 80]: #later [40, 80]
            for type in preprocessing_type:
                metrics = {"time": 0, "ram": [0, 0], "FP": 0, "FN": 0, "TP": 0, "Acc": 0, "curve_acc": 0, "modelName": modelName,"preprocessing":preprocessing_type,"size":size}
                try:
                    dataset_path = "C:/Users/adrie/Downloads/Lane_Detection/Datasets/TUSimple"
                    train_set, val_set, test_set = get_training_set(modelName, type, size, dataset_path)

                    #test_model = "C:/Users/adrie/PycharmProjects/LaneFinder1/TuSimple/80_80_files/UFDL/old_mod/goat.h5"
                    test_model = "C:/Users/adrie/PycharmProjects/LaneFinder1/TuSimple/80_80_files/U-Net/models/mobilenetv1_80_U-Net_1-no_sobel.h5"
                    model = load_model(test_model, compile=False)
                    #model = train(modelName, type, train_set, val_set) # TODO : make code work with custom train set and val_set

                    for img, label in train_set:

                        #preprocess(img, type) # TODO: or not because it's already done in the databese creation? is considered cheating?

                        prediction, ram, time = predict(model, modelName, img)
                        FP, FN, TP = accuracy(prediction, label, size, modelName)
                        curve_acc = curve_accuracy(prediction, label, size)

                        metrics["time"]+=time
                        metrics["ram"][0] += ram[0]
                        metrics["ram"][1] += ram[1]
                        metrics["FP"] += FP
                        metrics["FN"] += FN
                        metrics["TP"] += TP
                        metrics["curve_acc"] += curve_acc
                        #print("Used ram avg:", ram[0] / 1024, ", max:", ram[1] / 1024,"KB")


                    print("Used ram avg:", metrics["ram"][0]/len(train_set)/1024, ", max:", metrics["ram"][1]/len(train_set)/1024, "KB")
                    print("Average time:", metrics["time"]/len(train_set))
                    print("FP:", metrics["FP"]/len(train_set), " , FN:", metrics["FN"]/len(train_set), " , TP:", metrics["TP"]/len(train_set))
                    print("Curve error:", metrics["curve_acc"]/len(train_set))
                    try:
                        with open('C:/Users/adrie/Downloads/Lane_Detection/Evaluation/data.csv', 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(metrics)
                    except:
                        print("error saving csv:",metrics)

                except:
                    print("error occured", metrics)