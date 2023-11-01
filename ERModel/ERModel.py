from ERModel.statics.Config import PICKLED_ER_MODEL_PATH, TFIDF_WEIGHT, NAIVE_BAYES_WEIGHT
from .models.document import Document as Doc
from .NaiveBayes import NaiveBayes as NB
from .IO.Read import Reader
from .IO.Write import Writer
from .TFIDF import TFIDF


class ERM:
    """
        A Simple model for Emotion Recognition in Text which uses both TFIDF & Naive Bayes giving the result of each a weight.

    """
    def __init__(self):
        self.dataset = None
        self.emotion_set = None
        self.tfidf = TFIDF()
        self.naive_bayes = NB()
        return


    def train(self, train_dataset_path):
        """
            Trains the model on a new given dataset.

            input
                train_dataset_path: The path to the new dataset txt file to train on .
            
            returns 
                A copy of the model object after it is trained.
        """
        self.dataset = Reader.read_dataset(train_dataset_path)
        self._build_emotion_set()
        self.dataset = ERM._seperate_by_emotion(self.emotion_set, self.dataset)
        self._build_TFIDF_model()
        self._build_NB_model()
        return self


    def _build_emotion_set(self):
        """
            Goes through the dataset adding the emotion of each document to the emotion set.
        """
        self.emotion_set = set()
        for i, doc in enumerate(self.dataset):
            self.emotion_set.add(doc.emotion)
        return
    

    def _seperate_by_emotion(emotion_set, dataset):
        """
            Separates the dataset by their respective emotion adding each to 
            a dictionary the keys of which are the emotions from the emotion set.

            inputs
                emotion_set: The set of emotions/classes that the dataset holds.

                dataset: The dataset to be devided by emotions/classes into a dict.
            
            returns
                Returns a dictionary where the keys are the emotions/classes and 
                the values are the documentseach in their respective class.
        """
        res = dict([(emo, []) for emo in emotion_set])
        for val in emotion_set:
            for doc in dataset:
                if doc.emotion == val:
                    res[val].append(doc)
        return res

    def _build_TFIDF_model(self):
        """
            Trains the tfidf model on the dataset.
        """
        self.tfidf.train(self.dataset)
        return
    
    def _predict_TFIDF(self, text):
        """
            Predicts the emotion/class of an input text using the tfidf model.

            input
                text: The text to predict the class/emotion of.
            
            returns
                A dict of similarity of the text to each of the classes/emotions.
        """
        return self.tfidf.compare(text)
    
    def _build_NB_model(self):
        """
            Builds/trains the naive bayes model on the dataset.
        """
        self.naive_bayes.train(self.dataset)


    def _predict_NB(self, text):
        """
            Predicts the class/emotion of the input text usnig the naive bayes model.

            input
                text: The text to predict the class/emotion of.
            
            returns
                A dict of probability of the text being from each of the classes/emotions.
        """
        return self.naive_bayes.predict(text)
    

    def save_model(self):
        """
            Saves the ERM object into the file set in config.py.

            returns
                True if success False otherwise.
        """
        return Writer.write_pickled_obj(PICKLED_ER_MODEL_PATH, self)

    def load_model():
        """
            Loads an ERM model object from file set in the config.py.

            returns
                The ERM model object read from file.
        """
        return Reader.read_pickled_obj(PICKLED_ER_MODEL_PATH)


    def predict(self, text):
        """
            Predicts the probabilioty of the given text being in each of the emotions/classes.

            input
                text: The text to predict the class of.
            
            returns
                The final class/emotion which best represnts the given text along 
                with the similarity/probability of the text being from each of the emotions/classes.
        """
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
        """
            A method used to find the best value for tfidf&naive bayes weights.
            Requers change in class to be used.
            It is Not to be used as is.
        """
        self.NAIVE_BAYES_WEIGHT += 0.1
        self.TFIDF_WEIGHT = 1 - self.NAIVE_BAYES_WEIGHT