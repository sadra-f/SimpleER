from .models.document import Document as Doc
from .IO.Read import Reader
from .TFIDF import TFIDF

class ERM:
    def __init__(self, dataset_path):
        self.dataset = Reader.read_dataset(dataset_path)
        self._emotion_set = None
        self._build_emotion_set()
        tfidf = TFIDF(self.dataset)
        return


    def _build_emotion_set(self):
        self._emotion_set = set()
        for i, emotion in enumerate(self.dataset):
            self._emotion_set.add(emotion)
        return