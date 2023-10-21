from ERModel.statics.Config import  PICKLED_NAIVE_BAYES_PATH, PICKLED_TFIDF_PATH
from .models.document import Document as Doc
from .NaiveBayes import NaiveBayes as NB
from .IO.Read import Reader
from .TFIDF import TFIDF
import numpy as np
import pickle

class ERM:
    def __init__(self):
        self.dataset = None
        self.emotion_set = None
        self.tfidf = TFIDF()
        self.naive_bayes = NB()
        return


    def train(self, train_dataset_path):
        self.dataset = Reader.read_dataset(train_dataset_path)
        self._build_emotion_set()
        self._seperate_by_emotion()
        self._build_TFIDF_model()
        self._build_NB_model()
        return self


    def _build_emotion_set(self):
        self.emotion_set = set()
        for i, doc in enumerate(self.dataset):
            self.emotion_set.add(doc.emotion)
        return
    

    def _seperate_by_emotion(self):
        tmp = dict([(emo, []) for emo in self.emotion_set])
        for val in self.emotion_set:
            for doc in self.dataset:
                if doc.emotion == val:
                    tmp[val].append(doc)
        self.dataset = tmp

    def _build_TFIDF_model(self):
        self.tfidf.train([" ".join([val.string for val in self.dataset[emo]]) for emo in self.emotion_set])
        return
    
    def _predict_TFIDF(self, doc):
        return self.tfidf.compare(doc)
    
    def _build_NB_model(self):
        self.naive_bayes.train(self.dataset)


    def _predict_NB(self, text):
        return self.naive_bayes.predict(text)
    

    def save_model(self):
        with open(PICKLED_TFIDF_PATH, 'wb') as file:
            pickle.dump(self.tfidf, file)
        with open(PICKLED_NAIVE_BAYES_PATH, 'wb') as file:
            pickle.dump(self.naive_bayes, file)

    def load_model(self):
        with open(PICKLED_TFIDF_PATH, 'rb') as file:
            self.tfidf = pickle.load(file)
        with open(PICKLED_NAIVE_BAYES_PATH, 'rb') as file:
           self.naive_bayes = pickle.load(file)