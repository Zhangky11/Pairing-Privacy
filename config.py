import time

## set GLOBAL constants
DATASETS = ["credit", "census", "commercial"]
INITS = ["zero", "mean", "random"]
ABS_POSES = ["shap", "v"]
VFUNCS = ["l1", "l2", "log-odds"]

# the number of sampling used
USED_SAMPLE_NUM = {
    "credit": 500,
    "census": 1000,
    "commercial": 1000,
    "bike": 1000
} # must be an even number

# splitting ratio
SPLIT_RATIO = {
    "credit": 0.2,
    "census": 0.2,
    "commercial": 0.2,
    "bike": 0.2
}

# the maximum number of iterations when debug mode
MAX_NUM_ITR_DEBUG = 2

# when plot multiple lines, the maximum number of lines within one figure.
MAX_NUM_LINE = 25 

# the actuall numbe of lines within one graph.
ACTUAL_NUM_LINE = 10 

# parameter for clamp, train_max = X_train.mean + a * X_train.std, here we aims to set "a"
DEVIATION_NUM = 0.3

# the order
LOW_ORDER = 0.5

# Small value
EXTREME_SMALL_VAL = 1e-15

# Training parameters
TRAIN_ARGS = {
    "credit": {
        "if_fix": True,
        "model_seed": 9,
        "logspace": 1,
        "batch_size": 200,
        "train_lr": 0.01,
        "epoch": 500,
        "in_dim": 20,
        "hidd_dim": 100,
        "out_dim": 2
    }, 
    "census": {
        "if_fix": True,
        "model_seed": 2,
        "logspace": 0,
        "batch_size": 512,
        "train_lr": 0.1,
        "epoch": 1000,
        "in_dim": 12,
        "hidd_dim": 100,
        "out_dim": 2
    },
    "commercial": {
        "if_fix": True,
        "model_seed": 2,
        "logspace": 1,
        "batch_size": 512,
        "train_lr": 1e-5,
        "epoch": 400,
        "in_dim": 10,
        "hidd_dim": 100,
        "out_dim": 2
    }
}

DATE = time.strftime("%Y-%m-%d", time.localtime())
MOMENT = time.strftime("%H:%M:%S", time.localtime())