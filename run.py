from ERModel.ERModel import ERM
from ERModel.statics.Config import *



m = ERM(TRAIN_DATASET_PATH).train()

m._build_naive_bayes_model()