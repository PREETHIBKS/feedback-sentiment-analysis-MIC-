# feedback-sentiment-analysis-MIC-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
file=pd.read_csv('IMDB Dataset.csv')
x=file['review']
y=file['sentiment']
vectorizer=TfidfVectorizer(stop_words='english')
x_vect=vectorizer.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x_vect,y,test_size=0.2,random_state=40)
model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
y_prediction=model.predict(x_test)
print("Accuracy:\n",accuracy_score(y_test,y_prediction))
new_review=input('enter review')
new_vect=vectorizer.transform([new_review])
print('prediction:',model.predict(new_vect)[0])
