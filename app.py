# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:22:23 2021

@author: Sundar
"""
from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pandas as pd 
import pickle

#load the model
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))

app = Flask(__name__)
#model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def main():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['conversation']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('home.html',prediction = my_prediction)


if __name__ == '__main__':
    app.run()
    #app.run(host='127.0.0.1', port=9999, debug=True, use_reloader=False)
    
#	df= pd.read_csv("spam.csv", encoding="latin-1")
#	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
#	# Features and Labels
#	df['label'] = df['class'].map({'ham': 0, 'spam': 1})
#	X = df['message']
#	y = df['label']
#	
#	# Extract Feature With CountVectorizer
#	cv = CountVectorizer()
#	X = cv.fit_transform(X) # Fit the Data
#    
#    pickle.dump(cv, open('tranform.pkl', 'wb'))
#    
#    
#	from sklearn.model_selection import train_test_split
#	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#	#Naive Bayes Classifier
#	from sklearn.naive_bayes import MultinomialNB
#
#	clf = MultinomialNB()
#	clf.fit(X_train,y_train)
#	clf.score(X_test,y_test)
#    filename = 'nlp_model.pkl'
#    pickle.dump(clf, open(filename, 'wb'))
    
	#Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)    
    

