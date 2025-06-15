import enum
import importlib.util
import sys
from pathlib import Path


def main(task, size, model, params,
         dataset_path = None,
         train_set = None, val_set = None,
         trained_model = None,
         input_image = None):
    if task == "dataset":
        if dataset_path == None:
            raise ValueError("dataset selected with no path to dataset")
    elif task == "predict":
        if trained_model == None:
            raise ValueError("prediction selected with no model")
        if input_image == None:
            raise ValueError("prediction selected with no input image")
    elif task == "train":
        if train_set == None:
            raise ValueError("train selected with incorrect input train set")
        if val_set == None:
            raise ValueError("train selected with incorrect input validation set")
    else:
        raise ValueError("Not a valid function mode input")
    if size not in [80,40]:
        raise ValueError("Not a valid image size")
    if model not in  ["U-Net", "UFDL"]:
        raise ValueError("Not a valid data mode input")
    if params not in ["NO_SOBEL", "SOBEL_BLURRED", "SOBEL_SAMPLED", "SOBEL_MAX", "DUAL"]:
        raise ValueError("Not a valid data mode input")

    size_path = {40: "40_40_files", 80: "80_80_files"}[size]





def main(input = (2, 80, 0, 0)):
    if type(input) == type(()):
        raise ValueError("Input is not a tuple")
    if len(input) != 4:
        raise ValueError("Expected tuple of size 4")
    taskS = ["dataset", "predict", "train", "view_label"]
    sizeS = {40: "40_40_files", 80: "80_80_files"}  # in pixels
    modelS = ["U-Net", "UFDL"]  # CNN not implemented yet
    paramsS = ["NO_SOBEL", "SOBEL_BLURRED", "SOBEL_SAMPLED", "SOBEL_MAX", "DUAL"]


    task = taskS[input[0]]
    size = sizeS[input[1]]
    model = modelS[input[2]]
    params = paramsS[input[3]]



    # ---------- Locate Script ----------
    base_path = Path(__file__).parent
    module_path = base_path / size / model / f"{task}.py"

    # ---------- Load and Run ----------
    spec = importlib.util.spec_from_file_location("task_module", module_path)
    task_module = importlib.util.module_from_spec(spec)
    sys.modules["task_module"] = task_module
    spec.loader.exec_module(task_module)

    if hasattr(task_module, "main") and callable(task_module.main):
        result = task_module.main(params)  # now passing a simple string
    else:
        raise AttributeError(f"{module_path} does not have a callable 'main(params)' function.")

    return result