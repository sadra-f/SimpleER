# SimpleER

A simple Emotion Recognition model using tfidf & cosine similarity along with naive bayes.
Ready Pickled model is in the repo use the load method in the ERModel class to read them and then test/predict for new text using the predict method.

## Evaluations
using the evaluator class the latest evaluation results are :  
accuracy: 0.823  
precision:0.7761  
recall:0.7262  
f-score:0.7503  


## Sample Code

#### Train New Model
```
    from ERModel.ERModel import ERM
    from ERModel.statics.Config import *
    from ERModel.Evaluate import Evaluator

    m = ERM()
    m.train(TRAIN_DATASET_PATH)
    m.save_model()
```

#### Use Existing Model
```
    from ERModel.ERModel import ERM
    from ERModel.statics.Config import *
    from ERModel.Evaluate import Evaluator
    
    m = ERM()
    m = ERM.load_model()
    m.predict('today is a good day and that makes me feel great')
    eval = Evaluator(m, TEST_DATASET_PATH)
    evals = eval.evaluate()
    print(evals)
```

## Dataset

The dataset used is one from kaggle, a dataset for nlp purposes. [link](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)

Other datasets similar in structure can be used to train new models.
Similar to this structure:

A length of text;class
Second length of text;class

Where the class is the class/emotion which the text before the semicolon is represented by.


## Config file

The config.py file holds the paths to files used such as the log files and the datasets along with the Weight values for tfidf and naive bayes when predicting the outcome of the model the optimum values found for accuracy by experementing was NB=0.4,TFIDF=0.6 and almost similar NB=TFIDF = 0.5  
  
nb:0.2 tfidf:0.8 ==> 0.7885  
nb:0.3 tfidf:0.7 ==> 0.7995  
nb:0.4 tfidf:0.6 ==> 0.823  
nb:0.5 tfidf:0.5 ==> 0.8225  
nb:0.6 tfidf:0.4 ==> 0.8185  
nb:0.7 tfidf:0.3 ==> 0.811  
nb:0.8 tfidf:0.2 ==> 0.8035  