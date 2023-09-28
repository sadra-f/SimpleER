from .models.document import Document as Doc
import re
from .statics.Stopwords import STOP_WORDS
import numpy as np
from math import log10

class TFIDF:
    def __init__(self, docs:list[Doc]) -> None:
        self._docs = docs
        self.documents = docs
        self.terms = None
        self._tf = None
        self.idf = None
        self.tfidf = None
        # self._extract_documents()
        self.terms = self._extract_terms(self.documents)
        self.calculate_tf()
        self.calculate_idf()
        self.calculate_tfidf()

    def _extract_documents(self):
        self.documents = []
        for value in self._docs:
            self.documents.append(value.string)
        return


    def _extract_terms(self, documents):
        terms = set()
        for doc in documents:
            for term in re.split(' ', doc):
                terms.add(term)
        terms.difference_update(STOP_WORDS)
        terms = list(terms)
        return terms

    
    def calculate_tf(self):
        #tf -> term\document
        self._tf = np.ndarray((len(self.terms), len(self.documents)), np.float64)
        for i, trm in enumerate(self.terms):
            for j, doc in enumerate(self.documents):
                self._tf[i][j] = len(re.findall(f" {trm} ", doc))
        self.tf = np.add(self._tf, np.ones_like(self._tf))
        self.tf = np.log10(self.tf)
        # self.tf = np.add(self.tf, np.ones_like(self.tf), where=self.tf!=0)
        self.tf[self.tf != 0] += 1
        return



    def calculate_idf(self):
        self.idf = np.sum(np.ones_like(self._tf), 1, where=self._tf > 0)
        self.idf = np.divide(len(self.documents), self.idf)
        self.idf = np.log10(self.idf)
        self.idf = np.reshape(self.idf, (len(self.idf), 1))
        return
        
        


    def calculate_tfidf(self):
        self.tfidf = np.multiply(self._tf, self.idf)
        return
    

    def compare(self, doc):
        pass