from ERModel.ERModel import ERM
from ERModel.statics.Config import *
from ERModel.Evaluate import Evaluator
from datetime import datetime

m = ERM()
# m.train(TRAIN_DATASET_PATH)
# m.save_model()
m = ERM.load_model()
m.predict('today is a good day and that makes me feel great')
eval = Evaluator(m, TEST_DATASET_PATH)
evals = eval.evaluate()
print(evals)