from .models.document import Document as Doc
from .IO.Read import Reader
from .TFIDF import TFIDF
import numpy as np

class ERM:
    def __init__(self, dataset_path):
        self.dataset = Reader.read_dataset(dataset_path)
        self._emotion_set = None
        self._build_emotion_set()
        self.tfidf = TFIDF(self.dataset)
        self._seperate_by_emotion()
        self.seperated_tfidf = None
        return


    def _build_emotion_set(self):
        self._emotion_set = set()
        for i, doc in enumerate(self.dataset):
            self._emotion_set.add(doc.emotion)
        return
    

    def _seperate_by_emotion(self):
        tmp = dict([(emo, []) for emo in self._emotion_set])
        for val in self._emotion_set:
            for doc in self.dataset:
                if doc.emotion == val:
                    tmp[val].append(doc)
        self.dataset = tmp

    