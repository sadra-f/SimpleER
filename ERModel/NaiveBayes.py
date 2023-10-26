from ERModel.statics.Stopwords import STOP_WORDS
import numpy as np
import re

class NaiveBayes:
    """
        The Naive Bayes class which takes in a new dataset building a model which 
        can predict a new queries class/Emotion using the built model of the 
        probabilities of each of the words happening in each class

        fields:
            classes: The classes that the dataset can have
            documents: The list of documents in the dataset
            terms: The list of distinct terms in the dataset
            _ref: A reference dict, similar to the structure of the input dataset
            class_prob: A dict containg the probabiltity of each of the classes/emotions happening
            prob_dict: A dict of words and their probability of showing up for each class/emotion
    """
    def __init__(self):
        self.classes = None
        self.documents = None
        self.terms = None
        self._ref = None
        self.class_prob = None
        self.prob_dict = None
        

    def train(self, doc_dict:dict):
        """
            Trains/Builds the model using the input dataset.
            Calculates the probability of each document/sentence belonging to each of the classes/emotions.
            Calculates the probability of each term happening in each of the classes.
            
            input:
                doc_dict: The dataset in the form where keys are emotions/classes and the values 
                are lists of documents/sentences in/with that class/emotion.
            
            returns:
                None
        """
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
        """
            Calculates the porbability of the given inquery belonging to each of the classes/emotions.
            Does so by multiplying the probability of each class by the probability of each term in query being in that class.

            inputs:
                test: The document/sentecne to predict the class/emotion of.
            
            returns:
                The normalized probability of the input query belonging to each of the classes/emotions.
                A dict where the keys are the emotions/classes and the values are the probabilities
        """
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
        """
            Extracts the distinct terms that are in a list of documents/sentences.

            inputs:
                documents: The list of documents the terms of which we need to find.
            
            returns:
                A list of the distinct terms found within the documents
        """
        terms = set()
        for doc in documents:
            for term in re.split(' ', doc):
                terms.add(term)
        terms.difference_update(STOP_WORDS)
        terms = list(terms)
        return terms
    
    def min_max_normalizer(self, results:dict, new_min=0, new_max=1):
        """
            Normalizes the predictions of the model into a resonable scale from 0 to 1

            inputs:
                results: The probability dict holding the predictions for each class
                new_min: The new minimum value after the normalization
                new-max: the new maximum value after the normalization
            
            returns:    
                the normalized values for the input probability dict in the same dict
        """
        cur_min = min(results.values())
        cur_max = max(results.values())
        for key in results.keys():
            results[key] = ((results[key] - cur_min) / (cur_max - cur_min)) * (new_max - new_min) + new_min
        return results