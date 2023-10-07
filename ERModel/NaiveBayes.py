from .TFIDF import TFIDF
import numpy as np
from models.document import Document

class NaiveBayes:

    def __init__(self, doc_dict:dict):

        self.classes = list(doc_dict.keys())
        self.documents = list(doc_dict.values())
        self.terms = TFIDF._extract_terms(self.documents)

        self.term_document = np.full((len(self.documents), len(self.terms)), False, dtype=bool)
        for i, doc in enumerate(self.documents):
            for j, trm in enumerate(self.terms):
                self.term_document[i][j] = trm in doc

        self.class_prob = np.zeros((len(self.classes,2)))
        for i, value in enumerate(self.classes):
            self.class_prob[i][0] = value
            self.class_prob[i][1] = len(doc_dict[value]) / len(self.documents)


    

    def predict(self, text):
        results = np.zeros((len(self.classes),2))
        new_terms = TFIDF._extract_terms([text])
        for i, value in enumerate(self.classes):
            self.class_prob[i][0] = value
            prob = 0
            

        
