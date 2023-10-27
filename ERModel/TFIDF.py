from .CosineSimilarity import cosine_similarity
from .statics.Stopwords import STOP_WORDS
import numpy as np
import re

class TFIDF:
    """
        Builds a TFIDF model/matrix from a given dataset then predicts the probability of 
        a new query belonging to each of the classes/emotions using the term-document 
        vector from tfidf and cosine similarity.

        fields:
            documents: The documents being used to build the model over, in form of Document class objs in a dict where keys are classes/emotions.
            terms: The distinct terms in the train documents.
            tf: the normalized term frquency in the dataset documents.
            idf: The inverse document frequency of terms in the dataset documents.
            tfidf: The multiplicationof term frequency of terms in documetns with the inverse document frequency of terms in dataset.
            _ref_res: The reference dict for the results output.
    """
    def __init__(self):
        self._docs = None
        self.documents = None
        self.terms = None
        self.tf = None
        self.idf = None
        self.tfidf = None
        self._ref_res = None

    def train(self, docs:dict):
        """
            Train the tfidf model calculating the tfidf for the train dataset.

            inputs:
                docs: The train dataset, the tfidf of which will be calculated.
            
            returns:
                None
        """
        self._ref_res = dict()
        for class_name in docs:
            self._ref_res[class_name] = None

        self.classes = list(docs.keys())
        # self._docs = [' '.join([value.string for value in docs[class_name]]) for class_name in self.classes]
        self.documents = [' '.join([value.string for value in docs[class_name]]) for class_name in self.classes]
        # self._extract_documents()
        self.terms = TFIDF._extract_terms(self.documents)
        self._calculate_tf()
        self._calculate_idf()
        self._calculate_tfidf()

    def _extract_documents(self):
        """
            !DEPRECATED!
            Extracts the strings from the document objs in the _docs field into the documents field.
        """
        self.documents = []
        for value in self._docs:
            self.documents.append(value.string)
        return


    def _extract_terms(documents:list[str]):
        """
            Extracts the distinct terms that are in a list of documents/sentences.

            inputs:
                documents: The list of documents the terms of which we need to find.
            
            returns:
                A list of the distinct terms found within the documents.
        """
        terms = set()
        for doc in documents:
            for term in re.split(' ', doc):
                terms.add(term)

        terms.difference_update(STOP_WORDS)

        terms = list(terms)
        return terms

    
    def _calculate_tf(self):
        """
            Calculates the normalized term frequency of the terms in the train documents,counting the
            number of its occurrence in each of the documents by using log10 after counting the terms.
        """
        #tf -> term\document
        self.tf = np.ndarray((len(self.terms), len(self.documents)), np.float64)
        #count
        for i, trm in enumerate(self.terms):
            for j, doc in enumerate(self.documents):
                self.tf[i][j] = len(re.findall(f" {trm} ", doc))
        #normalize
        self.tf = np.add(self.tf, np.ones_like(self.tf))
        self.tf = np.log10(self.tf)
        # self.tf = np.add(self.tf, np.ones_like(self.tf), where=self.tf!=0)
        self.tf[self.tf != 0] += 1
        return



    def _calculate_idf(self):
        """
            Calcualte the inverse document frequency for each term counting the number of documents it has appeared on.
        """
        #count
        self.idf = np.sum(np.ones_like(self.tf), 1, where=self.tf > 0)
        #inverse
        self.idf = np.divide(len(self.documents), self.idf)
        #normalize
        self.idf = np.log10(self.idf)
        # self.idf = np.reshape(self.idf, (len(self.idf), 1))
        return
        
        


    def _calculate_tfidf(self):
        """
            Multiply the tf and idf matrcies.
        """
        self.tfidf = np.transpose(np.multiply(np.transpose(self.tf), self.idf))
        return
    

    def compare(self, new_doc:str):
        """
            Calculates the similarity/possibility of a new query belonging to each of the classes/emotions
            by building the tf of the new query multiplying it by the models idf and calculating the 
            cosine similarity of the new vector to the already existing vectors of the train documents.
        """
        #extract new terms.
        new_doc_terms = TFIDF._extract_terms([new_doc])
        #calc new query tf.
        new_doc_tf = np.zeros((len(self.terms)))
        for i, trm in enumerate(new_doc_terms):
            try:
                new_doc_tf[self.terms.index(trm)] = new_doc.count(f' {trm} ')
            except:
                pass

        new_doc_tf = np.add(new_doc_tf, np.ones_like(new_doc_tf))
        new_doc_tf = np.log10(new_doc_tf)
        new_doc_tf[new_doc_tf != 0] += 1
        #calc new doc tfidf multiply new doc tf and model idf.
        new_doc_tfidf = np.multiply(new_doc_tf, self.idf)
        #calc new query tfdif similarity of models existing tfidf for each class/emotion.
        result = self._ref_res.copy()
        for i, class_name in enumerate(self.classes):
            _sim = cosine_similarity(new_doc_tfidf, self.tfidf[:,i])
            result[class_name] = 0 if np.isnan(_sim) else _sim
        
        return self.min_max_normalizer(result)

    def min_max_normalizer(self, results:dict, new_min=0, new_max=1):
        """
            Normalizes the comparison results of cosine simlarity of new query and models existing vectors
            and bring the min and max of the results to the new max and min.

            inputs:
                results: The dict containing the similarity for each class/emotion.
                new_min: The new min after normalization.
                new_max: The new max after normalization.
        """
        cur_min = min(results.values())
        cur_max = max(results.values())
        for key in results.keys():
            try:
                results[key] = ((results[key] - cur_min) / (cur_max - cur_min)) * (new_max - new_min) + new_min
            except ZeroDivisionError:
                results[key] = 0
        return results