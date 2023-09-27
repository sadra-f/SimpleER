from .models.document import Document as Doc
from .IO.Read import Reader
from .TFIDF import TFIDF
import numpy as np

class ERM:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = Reader.read_dataset(self.dataset_path)
        self.emotion_set = None
        self.seperated_tfidf = None
        return


    def process(self):
        self._build_emotion_set()
        self._seperate_by_emotion()
        self._seperatly_calc_tfidf()
        return


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

    def _seperatly_calc_tfidf(self):
        self.seperated_tfidf = dict([(emo, None) for emo in self.emotion_set])
        for emo in self.emotion_set:
            self.seperated_tfidf[emo] = TFIDF(self.dataset[emo])
