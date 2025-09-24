# feedback-sentiment-analysis-MIC-tasks

##AIM:aims-to-built-a-machine-learning-model-that-analyzes-raw-text-data-from-movie-feedback-and-predicts-whether-the-sentiment-expressed-is-possitive-or-neagtive

##what-is-logistic-regression?
###Logistic-Regression-is-a-supervised-machine-learning-algorithm-used-for-classification-problems
###Unlike-linear-regression-which-predicts-continuous-values-it-predicts-the-probability-that-an-input-belongs-to-a-specific-class
###It-is-used-for-binary-classification-where-the-output-can-be-one-of-two-possible-categories-such-as-Yes/No-True/False-0/1
###It-uses-sigmoid-function-to-convert-inputs-into-a-probability-value-between-0and1

##what-is-text-vectorization?
###In-Natural-Language-Processing(NLP)-vectors-play-an-important-role-in-transforming-human-language-into-a-format-that-machines-can-comprehend-and-process
###These-numerical-representations-enable-computers-to-perform-tasks-such-as-sentiment-analysis-machine-translation-and-information-retrieval-with-greater-accuracy-and-efficiency

##TASK: It-involves-reading-raw-textual-feedback-data-from-an-Excel-file-using-Pandas-The-text-data-is-then-vectorized-using-TF_IDF-to-convert-it-into-numerical-features-suitable-for-machine-learning-A-LogisticRegression-model-is-trained-on-this-data-to-classify-movie-reviews-as-either-positive-or-negative

##importing-required-modules
###pandas-used-to-read-the-excel-file
import pandas as pd
###to-split-given-dataset-into-training-and-testing-sets
from sklearn.model_selection import train_test_split
###to-vectorize-the-raw-texts(many machine learning algorithm can't process raw tetual data)
from sklearn.feature_extraction.text import TfidfVectorizer
###Logistic-Rregression-is-classification-algorithm-that-predicts-categories(if positive/negative)
from sklearn.linear_model import LogisticRegression
###evaluation-tool-[percentage-of-correct-prediction]
from sklearn.metrics import accuracy_score

##main-codes
###loading-excel-file
file=pd.read_csv('IMDB Dataset.csv')
###reading-each-column
x=file['review']
y=file['sentiment']

###vectorize-raw-texts
vectorizer=TfidfVectorizer(stop_words='english')
x_vect=vectorizer.fit_transform(x)

###data-spliting
x_train,x_test,y_train,y_test=train_test_split(x_vect,y,test_size=0.2,random_state=40)

###train-logistic-regression
model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

###predictions
y_prediction=model.predict(x_test)

###evaluation
print("Accuracy:\n",accuracy_score(y_test,y_prediction))

###prediction-using-new-review
new_review=input('enter review')
new_vect=vectorizer.transform([new_review])
print('prediction:',model.predict(new_vect)[0])
