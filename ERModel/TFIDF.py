from .models.document import Document as Doc
import re
from .statics.Stopwords import STOP_WORDS
import numpy as np
from math import log10

class TFIDF:
    def __init__(self, docs:list[Doc]) -> None:
        self._docs = docs
        self.documents = None
        self.terms = None
        self.tf = None
        self.idf = None
        self.tfidf = None
        self._extract_documents()
        self._extract_terms()
        self.calculate_tf()
        self.calculate_idf()
        self.calculate_tfidf()

    def _extract_documents(self):
        self.documents = []
        for value in self._docs:
            self.documents.append(value.string)
        return

    def _extract_terms(self):
        self.terms = set()
        for doc in self.documents:
            for term in re.split(' ', doc):
                self.terms.add(term)
        self.terms.difference_update(STOP_WORDS)
        self.terms = list(self.terms)
        return



    def calculate_tf(self):
        #tf -> term\document
        self.tf = np.ndarray((len(self.terms), len(self.documents)), np.float64)
        for i, trm in enumerate(self.terms):
            for j, doc in enumerate(self.documents):
                self.tf[i][j] = doc.count(trm)
        self.tf = np.add(1,self.tf, where=self.tf!=0)
        self.tf = np.log10(self.tf, where=self.tf!=0)
        return



    def calculate_idf(self):
        self.idf = np.sum(self.tf, 0)
        self.idf = np.divide(len(self.documents), self.idf)
        self.idf = np.log10(self.idf)
        return
        
        


    def calculate_tfidf(self):
        self.tfidf = np.multiply(self.tf, self.idf)
        return