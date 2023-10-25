from .IO.Read import Reader
from .ERModel import ERM

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
        true_pos = self._ref_dict.copy()
        for key in self.testset.keys():
            for doc in self.testset[key]:
                test_counter += 1
                test_emo_counter[doc.emotion] += 1
                res = self.ermodel.predict(doc.string)
                predict_emo_counter[res[0]] += 1
                if res[0] == doc.emotion: 
                    true_counter += 1
                    true_pos[key] += 1
        
        return EvaluationResult(test_counter, true_counter, test_emo_counter, true_pos, predict_emo_counter)

            
class EvaluationResult:
    def __init__(self, total_count, true_count, test_counts:dict, true_predict:dict, predict_counts:dict) -> None:
        self.accuracy = true_count / total_count
        self.recall = test_counts.copy()
        self.total_recall = 0
        self.precision = test_counts.copy()
        self.total_precision = 0
        for key in test_counts.keys():
            self.recall[key] = true_predict[key] / test_counts[key]
            self.total_recall += self.recall[key]
            self.precision[key] = true_predict[key] / predict_counts[key]
            self.total_precision += self.precision[key]
        self.total_recall /= len(test_counts.keys())
        self.total_precision /= len(test_counts.keys())
        self.f_score = (2 * self.total_recall * self.total_precision) / (self.total_precision + self.total_recall)

    def __str__(self) -> str:
        return f'accuracy: {self.accuracy}\r\nprecision:{self.total_precision}\r\nrecall:{self.total_recall}\r\nf-score:{self.f_score}'