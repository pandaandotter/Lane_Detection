models = []
preprocessing_type = []
import tracemalloc
from Run_Test import loader
import numpy as np
import collections

def train(modelName, type, model_size, train_set, val_set):
    return loader((2,model_size,modelName,type))
#def preprocess(img, mode):
#    a=1
def predict(model_loc, modelName, img):
    #return loader((2,model_size,modelName,type)) # TODO: insert way to include
    if modelName == "U-net":   #TODO: insert UFDL/
        return model.predict(img)[0, :, :, 0]  # (40, 40) #U-net
    if modelName == "UFDL":
        outputs = model.predict(img)
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
        return presence, col_bins

    # Get the most likely bin index per row
    presence = np.array(lane_exist[0]).flatten()
    col_bins = np.argmax(lane_pos[0], axis=-1)

    counter = collections.Counter(np.atleast_1d(col_bins))

def accuracy(prediciton, label):
    return "bad", "badder" # TODO: evaluate accurecy
def curve(label):
    return "curve"
def compare_curves(curve1, curve2):
    return curve1-curve2 #abs is ideal, maybe a square... who knows
def curve_accuracy(prediciton, label):
    return compare_curves(curve(label),curve(prediciton))

for modelName in models:
    for size in [80, 40]:
        for type in preprocessing_type:
            train_set, val_set, test_set = [],[],[]   # TODO
            model = train(modelName, type, train_set, val_set) # TODO : make code work with custom train set and val_set
            for img, label in train_set:
                #preprocess(img, type) # TODO: or not because it's already done in the databese creation? is considered cheating?
                tracemalloc.clear_traces()
                tracemalloc.start()#ram_measure_start()
                prediciton = predict(model, img)
                current, peak = tracemalloc.get_traced_memory()#res_ram = ram_measure_end()
                TP,FN = accuracy(prediciton, label)
                curve_acc = curve_accuracy(prediciton, label)


