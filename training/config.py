"""
This file defines default options of configurations
It will be further merged by yaml files and options from
the command-line
Note that *any* hyper-parameters should be firstly defined
here to enable yaml and command-line configuration
"""

_C = dict()

# Logging and saving
_C["LOG"] = dict()
_C["LOG"]["SAVE_FREQ"] = 5
_C["LOG"]["VIS_FREQ"] = 1

# Data settings
_C["DATA"] = dict()
_C["DATA"]["PATH"] = './data/MT-Dataset'
_C["DATA"]["NUM_WORKERS"] = 4
_C["DATA"]["BATCH_SIZE"] = 1
_C["DATA"]["IMG_SIZE"] = 256

# Training hyper-parameters
_C["TRAINING"] = dict()
_C["TRAINING"]["G_LR"] = 2e-4
_C["TRAINING"]["D_LR"] = 2e-4
_C["TRAINING"]["BETA1"] = 0.5
_C["TRAINING"]["BETA2"] = 0.999
_C["TRAINING"]["NUM_EPOCHS"] = 50
_C["TRAINING"]["LR_DECAY_FACTOR"] = 5e-2
_C["TRAINING"]["DOUBLE_D"] = False
_C["TRAINING"]["DECAY_SKIN"] = False # True
_C["TRAINING"]["DECAY_LIP_EYE"] = False # True

# Loss weights
_C["LOSS"] = dict()
_C["LOSS"]["LAMBDA_A"] = 10.0
_C["LOSS"]["LAMBDA_B"] = 10.0
_C["LOSS"]["LAMBDA_IDT"] = 0.5
_C["LOSS"]["LAMBDA_REC"] = 10
_C["LOSS"]["LAMBDA_MAKEUP"] = 100
_C["LOSS"]["LAMBDA_SKIN"] = 0.1
_C["LOSS"]["LAMBDA_EYE"] = 1.5
_C["LOSS"]["LAMBDA_LIP"] = 1
_C["LOSS"]["LAMBDA_MAKEUP_LIP"] = 100 # _C["LOSS["LAMBDA_MAKEUP"]"] * _C["LOSS["LAMBDA_LIP"]"]
_C["LOSS"]["LAMBDA_MAKEUP_SKIN"] = 10 # _C["LOSS["LAMBDA_MAKEUP"]"] * _C["LOSS["LAMBDA_SKIN"]"]
_C["LOSS"]["LAMBDA_MAKEUP_EYE"] = 150 # _C["LOSS["LAMBDA_MAKEUP"]"] * _C["LOSS["LAMBDA_EYE"]"]
_C["LOSS"]["LAMBDA_VGG"] = 5e-3
# Added
_C["LOSS"]["LAMBDA_NO_MAKEUP"] = 1 # 1
_C["LOSS"]["LAMBDA_CYCLE_LOSS_VERSION"] = 1

# Model structure
_C["MODEL"] = dict()
_C["MODEL"]["D_TYPE"] = 'SN'
_C["MODEL"]["D_REPEAT_NUM"] = 3
_C["MODEL"]["D_CONV_DIM"] = 64
_C["MODEL"]["G_CONV_DIM"] = 64
_C["MODEL"]["NUM_HEAD"] = 1
_C["MODEL"]["DOUBLE_E"] = True # True
_C["MODEL"]["DOUBLE_DEC"] = True # True
_C["MODEL"]["USE_FF"] = False
_C["MODEL"]["NUM_LAYER_E"] = 3
_C["MODEL"]["NUM_LAYER_D"] = 2
_C["MODEL"]["WINDOW_SIZE"] = 16
_C["MODEL"]["MERGE_MODE"] = 'conv'

# Preprocessing
_C["PREPROCESS"] = dict()
_C["PREPROCESS"]["UP_RATIO"] = 0.6 / 0.85  # delta_size / face_size
_C["PREPROCESS"]["DOWN_RATIO"] = 0.2 / 0.85  # delta_size / face_size
_C["PREPROCESS"]["WIDTH_RATIO"] = 0.2 / 0.85  # delta_size / face_size
_C["PREPROCESS"]["LIP_CLASS"] = [7, 9]
_C["PREPROCESS"]["FACE_CLASS"] = [1, 6]
_C["PREPROCESS"]["EYEBROW_CLASS"] = [2, 3]
_C["PREPROCESS"]["EYE_CLASS"] = [4, 5]
_C["PREPROCESS"]["NECK_CLASS"] = [13]
_C["PREPROCESS"]["NOSE_CLASS"] = [6]
_C["PREPROCESS"]["LANDMARK_POINTS"] = 68

# Pseudo ground truth
_C["PGT"] = dict()
_C["PGT"]["EYE_MARGIN"] = 12
_C["PGT"]["LIP_MARGIN"] = 4
# Added
_C["PGT"]["TYPE"] = 3 # 3
_C["PGT"]["SKIN_ALPHA"] = 0.3
_C["PGT"]["SKIN_ALPHA_MILESTONES"] = (0, 12, 24, 50)
_C["PGT"]["SKIN_ALPHA_VALUES"] = (0.2, 0.4, 0.3, 0.2)
_C["PGT"]["EYE_ALPHA"] = 0.8
_C["PGT"]["EYE_ALPHA_MILESTONES"] = (0, 12, 24, 50)
_C["PGT"]["EYE_ALPHA_VALUES"] = (0.6, 0.8, 0.6, 0.4)
_C["PGT"]["LIP_ALPHA"] = 0.1
_C["PGT"]["LIP_ALPHA_MILESTONES"] = (0, 12, 24, 50)
_C["PGT"]["LIP_ALPHA_VALUES"] = (0.05, 0.2, 0.1, 0.0)

# Postprocessing
_C["POSTPROCESS"] = dict()
_C["POSTPROCESS"]["WILL_DENOISE"] = False


# Collaborator
if _C["PGT"]["TYPE"] == 3:
    _C["TRAINING"]["C_LR"] = 2e-4
    _C["TRAINING"]["D_C_LR"] = 2e-4
    _C["LOSS"]["LAMBDA_C_GT"] = 1
    _C["LOSS"]["LAMBDA_C_D"] = 0.1

def get_config()->dict:
    return _C
