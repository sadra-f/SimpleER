from ERModel.ERModel import ERM
from ERModel.statics.Config import *



m = ERM(TRAIN_DATASET_PATH).train()

m._predict_cosine_sim("i feel really sad because i was damaged")