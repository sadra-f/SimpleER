from .ERModel import ERM
from .IO.Read import Reader
import numpy as np

class Evaluator:
    def __init__(self, ermodel:ERM, testpath=None):
        self.ermodel = ermodel
        self.classes = list(ermodel.emotion_set)
        self._ref_dict = dict()
        for value in self.classes: self._ref_dict[value] = 0
        self.testset = None
        self.testpath = testpath
        if testpath != None:
            self.load_test_dataset(testpath)


    def load_test_dataset(self, path):
        self.testpath = path
        self.testset = Reader.read_dataset(self.testpath)
        self.testset = ERM._seperate_by_emotion(self.classes, self.testset)
    

    def evaluate(self):
        test_counter = 0
        true_counter = 0
        test_emo_counter = self._ref_dict.copy()
        predict_emo_counter = self._ref_dict.copy()
        for key in self.testset.keys():
            for doc in self.testset[key]:
                test_counter += 1
                test_emo_counter[doc.emotion] += 1
                res = self.ermodel.predict(doc.string)
                predict_emo_counter[res[0]] += 1
                if res[0] == doc.emotion: true_counter += 1
        
        return true_counter / test_counter

            
