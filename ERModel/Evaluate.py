from .IO.Read import Reader
from .ERModel import ERM

class Evaluator:
    """
        Evaluator class measures several metrics that provides info on the performance of he predicting model

        fields:
            ermodel: the EmotionRecognition model to measure the performance of
            classes: the list of the emotions/classes the dataset can have
            _ref_dict: The reference dictionary that holds the structure wanted for ouput
            testset: The test dataset used to evaluate the model
            testpath: The path to the test dataset
    """
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
        """
            Reads the test dataset from the testpath and seperates the documents within 
            a dictionary where the keys are the classes/emotion and the values are the documents

            inputs:
                path: The path to the test dataset
        """
        self.testpath = path
        self.testset = Reader.read_dataset(self.testpath)
        self.testset = ERM._seperate_by_emotion(self.classes, self.testset)
    

    def evaluate(self):
        """
            Evaluates the performance of the model using the test dataset

            returns:
                Metrics showing how well the model has performed including metrics such as accuracy, recall, precision and f-score
        """
        #init the values
        test_counter = 0
        true_counter = 0
        test_emo_counter = self._ref_dict.copy()
        predict_emo_counter = self._ref_dict.copy()
        true_pos = self._ref_dict.copy()
        #use model to predict and count True/False predictions
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
    """
        Evaluation result holder class which takes in certain values regarding the prediction results and 
        returns evaluative parameters such asaccuracy, recall, precision and f-score.
    """
    def __init__(self, total_count, true_count, test_counts:dict, true_predict:dict, predict_counts:dict) -> None:
        #accuracy
        self.accuracy = true_count / total_count
        #init recall for each class and overall recall
        self.recall = test_counts.copy()
        self.total_recall = 0
        #init precision for each class/emotion and overal precision
        self.precision = test_counts.copy()
        self.total_precision = 0
        #
        for key in test_counts.keys():
            #calc recall/overall recall
            self.recall[key] = true_predict[key] / test_counts[key]
            self.total_recall += self.recall[key]
            #calc precision/overall precision
            self.precision[key] = true_predict[key] / predict_counts[key]
            self.total_precision += self.precision[key]

        #final calculation of total recall&precision
        self.total_recall /= len(test_counts.keys())
        self.total_precision /= len(test_counts.keys())
        #f-score calculation
        self.f_score = (2 * self.total_recall * self.total_precision) / (self.total_precision + self.total_recall)

    def __str__(self) -> str:
        return f'accuracy: {self.accuracy}\r\nprecision:{self.total_precision}\r\nrecall:{self.total_recall}\r\nf-score:{self.f_score}'