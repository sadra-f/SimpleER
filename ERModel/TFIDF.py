from .models.document import Document as Doc
import re
from .statics.Stopwords import STOP_WORDS
import numpy as np
from math import log10
from .CosineSimilarity import cosine_similarity

class TFIDF:
    def __init__(self):
        self._docs = None
        self.documents = None
        self.terms = None
        self._tf = None
        self.idf = None
        self.tfidf = None

    def train(self, docs:list[Doc]):
        self._docs = docs
        self.documents = docs
        # self._extract_documents()
        self.terms = TFIDF._extract_terms(self.documents)
        self.calculate_tf()
        self.calculate_idf()
        self.calculate_tfidf()

    def _extract_documents(self):
        self.documents = []
        for value in self._docs:
            self.documents.append(value.string)
        return


    def _extract_terms(documents:list[str]):
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
        # self.idf = np.reshape(self.idf, (len(self.idf), 1))
        return
        
        


    def calculate_tfidf(self):
        self.tfidf = np.transpose(np.multiply(np.transpose(self._tf), self.idf))
        return
    

    def compare(self, new_doc:str):
        new_doc_terms = TFIDF._extract_terms([new_doc])
        new_doc_tf = np.zeros((len(self.terms)))
        for i, trm in enumerate(new_doc_terms):
            new_doc_tf[self.terms.index(trm)] = new_doc.count(trm)
        new_doc_tf = np.add(new_doc_tf, np.ones_like(new_doc_tf))
        new_doc_tf = np.log10(new_doc_tf)
        new_doc_tf[new_doc_tf != 0] += 1
        new_doc_tfidf = np.multiply(new_doc_tf, self.idf)
        result = np.zeros((1,len(self.documents)))
        for i, document in enumerate(self.documents):
            result[0][i] = cosine_similarity(new_doc_tfidf, self.tfidf[:,i])
        
        return result

