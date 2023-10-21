from ERModel.ERModel import ERM
from ERModel.statics.Config import *



m = ERM()
m.train(TRAIN_DATASET_PATH)
m.save_model()
# m.load_model()
res = m.predict("i am very happy today")
print('done')