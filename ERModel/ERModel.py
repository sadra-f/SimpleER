from ERModel.statics.Config import PICKLED_ER_MODEL_PATH, TFIDF_WEIGHT, NAIVE_BAYES_WEIGHT
from .IO.Read import Reader, read_pickled_obj
from .models.document import Document as Doc
from .NaiveBayes import NaiveBayes as NB
from .IO.Write import write_pickled_obj
from .TFIDF import TFIDF


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
        self.dataset = ERM._seperate_by_emotion(self.emotion_set, self.dataset)
        self._build_TFIDF_model()
        self._build_NB_model()
        return self


    def _build_emotion_set(self):
        self.emotion_set = set()
        for i, doc in enumerate(self.dataset):
            self.emotion_set.add(doc.emotion)
        return
    

    def _seperate_by_emotion(emotion_set, dataset):
        res = dict([(emo, []) for emo in emotion_set])
        for val in emotion_set:
            for doc in dataset:
                if doc.emotion == val:
                    res[val].append(doc)
        return res

    def _build_TFIDF_model(self):
        self.tfidf.train(self.dataset)
        return
    
    def _predict_TFIDF(self, text):
        return self.tfidf.compare(text)
    
    def _build_NB_model(self):
        self.naive_bayes.train(self.dataset)


    def _predict_NB(self, text):
        return self.naive_bayes.predict(text)
    

    def save_model(self):
        write_pickled_obj(PICKLED_ER_MODEL_PATH, self)

    def load_model():
        return read_pickled_obj(PICKLED_ER_MODEL_PATH)


    def predict(self, text):
        if text is Doc:
            text = text.string
        tfidf = self._predict_TFIDF(text)
        nb = self._predict_NB(text)
        res = 'something went wrong'
        results = dict()
        largest_sim = -1
        for i, emotion in enumerate(self.emotion_set):
            emo_sim = TFIDF_WEIGHT * tfidf[emotion] + NAIVE_BAYES_WEIGHT * nb[emotion]
            if emo_sim > largest_sim :
                largest_sim = emo_sim
                res = emotion
            results[emotion] = emo_sim
        return (res, results)
    
    def _update_weights(self):
        self.NAIVE_BAYES_WEIGHT += 0.1
        self.TFIDF_WEIGHT = 1 - self.NAIVE_BAYES_WEIGHT