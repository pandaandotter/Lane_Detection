import enum
import importlib.util
import sys
from pathlib import Path


# dataset is no longer necessary, it is still here to check if it runs after some modifications
taskS = ["dataset", "predict","train","view_label"]
sizeS = {40: "40_40_files", 80 : "80_80_files" } # in pixels
modelS = ["U-Net", "UFDL"] # CNN not implemented yet
paramsS = ["NO_SOBEL","SOBEL_BLURRED","SOBEL_SAMPLED","SOBEL_MAX", "DUAL"]
# all the options are single channel input except for "DUAl" who will input a "NO_SOBEL" and a "SOBEL_BLURRED" as a 2 channel input


task = taskS[1]             # "dataset", "predict","train","view_label"
size = sizeS[40]            # 40 or 80
model = modelS[0]           # Model type, U-Net, UFDL... it was planned to have CNN and BNN too but the results from simple neural networks were unusable
params = paramsS[0]         # "NO_SOBEL","SOBEL_BLURRED","SOBEL_SAMPLED"...


# build total path with params
base_path = Path(__file__).parent
module_path =  base_path / size / model / f"{task}.py"

# load and run
spec = importlib.util.spec_from_file_location("task_module", module_path)
task_module = importlib.util.module_from_spec(spec)
sys.modules["task_module"] = task_module
spec.loader.exec_module(task_module)

if hasattr(task_module, "main") and callable(task_module.main):
    task_module.main(params)  # now passing a simple string
else:
    raise AttributeError(f"{module_path} does not have a callable 'main(params)' function.")
