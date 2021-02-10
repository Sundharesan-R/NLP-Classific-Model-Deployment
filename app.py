# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:22:23 2021

@author: Sundar
"""
from flask import Flask,render_template,url_for,request
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import pandas as pd 
import pickle

#load the model
filename = 'nlp_model.pkl'
classifier = pickle.load(open(filename, 'rb'))
vectorizer=pickle.load(open('tranform.pkl','rb'))

app = Flask(__name__)



@app.route('/')
def main():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['conversation']
        data = [message]
        vect = vectorizer.transform(data).toarray()
        my_prediction = classifier.predict(vect)
    return render_template('home.html',prediction = my_prediction)


if __name__ == '__main__':
    app.run()

    

