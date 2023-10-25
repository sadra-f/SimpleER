# SimpleER

A simple Emotion Recognition model using tfidf & cosine similarity along with naive bayes.
Ready models are in the repo use the load method in the ERModel class to read them and then test/predict for new text using the predict method.

## Evaluations
using the evaluator class the latest evaluation results are :
accuracy: 0.823
precision:0.7761
recall:0.7262
f-score:0.7503


### Sample Code

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