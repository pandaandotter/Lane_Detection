import enum
import importlib.util
import sys
from pathlib import Path


def run_task(task, size, model, params,
         dataset_path=None,
         train_set=None, val_set=None):

    if size not in [80, 40]:
        raise ValueError("Not a valid image size")
    if model not in ["U-Net", "UFDL"]:
        raise ValueError("Not a valid data mode input")
    if params not in ["NO_SOBEL", "SOBEL_BLURRED", "SOBEL_SAMPLED", "SOBEL_MAX", "DUAL"]:
        raise ValueError("Not a valid data mode input")

    size_path = {40: "40_40_files", 80: "80_80_files"}[size]
    base_path = Path(__file__).parent.resolve()
    base_path = base_path.parent
    module_path =  base_path / "ML" / size_path / model / f"{task}.py"

    spec = importlib.util.spec_from_file_location("task_module", module_path)
    task_module = importlib.util.module_from_spec(spec)
    sys.modules["task_module"] = task_module
    spec.loader.exec_module(task_module)

    if hasattr(task_module, "main") and callable(task_module.main):
        if task == "dataset":
            if dataset_path == None:
                raise ValueError("dataset selected with no path to dataset")
            result = task_module.main(params, DB_location=dataset_path)
        elif task == "train":
            if train_set == None:
                raise ValueError("train selected with incorrect input train set")
            if val_set == None:
                raise ValueError("train selected with incorrect input validation set")
            result = task_module.main(params, train_set, val_set)
        else:
            raise ValueError("Not a valid function mode input")
    else:
        raise AttributeError(f"{module_path} does not have a callable 'main(params)' function.")



    return result