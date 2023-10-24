from ERModel.statics.Stopwords import STOP_WORDS
import numpy as np
import re

class NaiveBayes:

    def __init__(self):
        self.classes = None
        self.documents = None
        self.terms = None
        self._ref = None
        self.class_prob = None
        self.prob_dict = None
        

    def train(self, doc_dict:dict):
        self.classes = list(doc_dict.keys())
        self.documents = list(np.concatenate([*doc_dict.values()]))
        self.terms = NaiveBayes._extract_terms([val.string for val in self.documents])

        self._ref = dict()
        for class_name in self.classes:
            self._ref[class_name] = None
        
        self.class_prob = self._ref.copy()
        self.prob_dict = self._ref.copy()

        for class_name in self.classes:
            self.class_prob[class_name] = len(doc_dict[class_name]) / len(self.documents)
            self.prob_dict[class_name] = np.ones((len(self.terms,)))
            _sent_joined = ' '.join([val.string for val in doc_dict[class_name]])
            for i, term in enumerate(self.terms):
                self.prob_dict[class_name][i] += _sent_joined.count(f' {term} ')
            self.prob_dict[class_name] = np.divide(self.prob_dict[class_name], np.sum(self.prob_dict[class_name], 0))

    def predict(self, test):
        results = self._ref.copy()
        test_terms = NaiveBayes._extract_terms([test])
        for class_name in self.classes:
            results[class_name] = self.class_prob[class_name]
            for new_term in test_terms:
                try:
                    results[class_name] *= self.prob_dict[class_name][self.terms.index(new_term)]
                except:
                    pass
        
        return self.min_max_normalizer(results)
    
    def _extract_terms(documents:list[str]):
        terms = set()
        for doc in documents:
            for term in re.split(' ', doc):
                terms.add(term)
        terms.difference_update(STOP_WORDS)
        terms = list(terms)
        return terms
    
    def min_max_normalizer(self, results:dict, new_min=0, new_max=1):
        cur_min = min(results.values())
        cur_max = max(results.values())
        for key in results.keys():
            results[key] = ((results[key] - cur_min) / (cur_max - cur_min)) * (new_max - new_min) + new_min
        return results