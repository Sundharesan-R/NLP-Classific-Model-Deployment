# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 23:09:28 2021

@author: Sundar
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import nltk
import re
import unicodedata
import tqdm
import collections
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report
np.set_printoptions(precision=2, linewidth=80)
from wordcloud import WordCloud, STOPWORDS
ps = nltk.porter.PorterStemmer()
ls = nltk.stem.LancasterStemmer()

train_data=pd.read_excel('NLP Engineer -Train&val Dataset.xlsx')
train_x=train_data["Conversations"]
train_y=train_data["Patient Or not"]

train_data["Patient Or not"].value_counts()

#Extract Email:
email = re.compile(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")
mail=email.findall(train_x.to_string())

'''def simple_stemmers(text,stemmer = ps):
    text = " ".join([stemmer.stem(word)for word in text.split()])
    return text'''


#Preprocessing
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}

def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    text = " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])
    return text

def remove_diacritics(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

contractions_dict = {
    'didn\'t': 'did not',
    'don\'t': 'do not',
    "aren't": "are not",
    "can't": "cannot",
    "cant": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "didnt": "did not",
    "doesn't": "does not",
    "doesnt": "does not",
    "don't": "do not",
    "dont" : "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i had",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'm": "i am",
    "im": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had",
    "she'd've": "she would have",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they had",
    "they'd've": "they would have",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
    }

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

def remove_stopwords(text, is_lower_case=False, stopwords=None):
    if not stopwords:
        stopwords = nltk.corpus.stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


def pre_process_corpus(docs):
    norm_docs = []
    for doc in tqdm.tqdm(docs):
#        doc = simple_stemmers(doc)
        doc = lemmatize_words(doc)
        doc = remove_urls(doc)
        doc = strip_html_tags(doc)
        doc = expand_contractions(doc)
        doc = remove_stopwords(doc)
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))
        doc = remove_diacritics(doc)
        doc = re.sub(r'[/+,!+,#+,%+,&+,-,_+,=+,:+,;+,\.+, \++, \*+, \?+, \^+, \$+, \(, \), \[+, \]+, \{+, \}+, \|+, \\+ \s]', ' ', doc, re.I|re.A)
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))
        doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
        doc = doc.lower()
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()
        norm_docs.append(doc)
    return norm_docs


c= pre_process_corpus(train_x)
text_freq=pd.DataFrame(c)
text_freq.columns=['Conversations']

text=pd.DataFrame(c)
text.columns=['Conversations']




from nltk.tokenize import sent_tokenize
text['split'] = text['Conversations'].apply(sent_tokenize)


from nltk.tokenize import word_tokenize
text['split_words'] = text['Conversations'].apply(word_tokenize)


#SMOTE-Imbalance-Oversampling:
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


vectorizer = CountVectorizer()   
X=vectorizer.fit_transform(c).toarray()

pickle.dump(vectorizer, open('tranform.pkl', 'wb'))

Y=np.array(train_y)

smt = SMOTE()
x_sm, y_sm = smt.fit_sample(X, Y)

tem=pd.DataFrame(y_sm)
tem[0].value_counts()



#Sentiment Analysis:

text_sentiment=pd.DataFrame(text['Conversations'])
text_sentiment['Patient Or not']=train_y

import textblob
def convert(text):
    if text == 1:
        return 'positive'
    if text == 0:
        return 'negative'
text_sentiment['sentiment'] = text_sentiment['Patient Or not'].apply(convert)

text_sentiment['sentiment'].value_counts()


conversations = np.array(text_sentiment['Conversations'])
sentiments = np.array(text_sentiment['sentiment'])
sample = [353,699,900]

for conversation, sentiment in zip(conversations[sample],sentiments[sample]):
    print("CONVERSATION:",conversation)
    print("SENTIMENT:",sentiment)
    print('Predicted Sentiment polarity:',textblob.TextBlob(conversation).sentiment.polarity)
    print('-'*60)
    
sentiment_polarity = [textblob.TextBlob(conversation).sentiment.polarity for conversation in conversations]
predicted_sentiments = ['positive' if score >= 0.1 else 'negative' for score in sentiment_polarity]    

labels = ['negative','positive']
print(classification_report(sentiments,predicted_sentiments))
pd.DataFrame(confusion_matrix(sentiments, predicted_sentiments), index = labels,columns = labels)



#Word Frequency Table:
tokens = re.findall('\w+',text_freq.to_string())
tokens = [i for i in tokens if not (i.isdigit() 
                                         or i[0] == '-' and i[1:].isdigit())]
tokens.remove('Conversations')    
sb.set(rc={'figure.figsize':(11.7,8.27)})    
sb.set_style('darkgrid')
nlp_words=nltk.FreqDist(tokens)
nlp_words.plot(20)
plt.show()

#Sentence Frequency Table:
tokens_sen = text_freq['Conversations']
sb.set(rc={'figure.figsize':(11.7,8.27)})    
sb.set_style('darkgrid')
nlp_sent=nltk.FreqDist(tokens_sen)
nlp_sent.plot(20)
plt.show()

#Features: 
#n-grams
from nltk.util import ngrams
esFourgrams = ngrams(tokens,4)

esFourgramsFreq = collections.Counter(esFourgrams)
esFourgramsFreq.most_common(10)

#Map words into Vectors
word_set = re.findall('\w+',text_freq.to_string())
word_set = [i for i in word_set if not (i.isdigit() 
                                         or i[0] == '-' and i[1:].isdigit())]
word_set.remove('Conversations')

wordDictA = dict.fromkeys(word_set,0)
for word in word_set:
    wordDictA[word]+=1
wordDict = pd.DataFrame([wordDictA])

#Visualization:
#Top 20
from nltk import FreqDist
freq = FreqDist(tokens)
top = freq.most_common(20)
x,y = zip(*top)
plt.subplots(figsize=(18,8))
plt.bar(x, y)

#Bottom 20
fd = FreqDist(tokens)
bottom = fd.most_common()[-20:]
x,y = zip(*bottom)
plt.subplots(figsize=(18,8))
plt.bar(x, y)

#WorCloud
stopwords = set(STOPWORDS)
comment_words = ''
for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 



#Summarization:
sentences = text['Conversations'].to_string()
result = ''.join([i for i in sentences if not i.isdigit()])
result = re.sub('[!@#$.]', '', result)
result = re.sub(r'[^a-zA-Z\s]', '', result, re.I|re.A)
result = re.sub(' +', ' ', result)
result = result.strip()

stopwords = nltk.corpus.stopwords.words('english')
word_frequencies = {}
for word in nltk.word_tokenize(result):
    if word not in stopwords:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1
            
maximum_frequncy = max(word_frequencies.values())

for word in word_frequencies.keys():
    word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)            

sentence_list = text['Conversations'].tolist()

sentence_scores = {}
for sent in sentence_list:
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_frequencies.keys():
            if len(sent.split(' ')) < 30:
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]
                    
import heapq
summary_sentences = heapq.nlargest(50, sentence_scores, key=sentence_scores.get)

summary = ' '.join(summary_sentences)
print(summary)                    


#Modeling & Tuning:

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

X_train,X_test,Y_train,Y_test=train_test_split(x_sm,y_sm,test_size=0.20,random_state=0)
pipeline_lr=Pipeline([('scalar1',StandardScaler()),
                     ('pca1',PCA(n_components=2)),('lr_classifier',LogisticRegression(random_state=0))])
model = pipeline_lr.fit(X_train, Y_train)
model.score(X_test,Y_test)

pipeline_lr=Pipeline([('scalar1',StandardScaler()),
                     ('pca1',PCA(n_components=2)), 
                     ('lr_classifier',LogisticRegression())])
pipeline_dt=Pipeline([('scalar2',StandardScaler()),
                     ('pca2',PCA(n_components=2)),
                     ('dt_classifier',DecisionTreeClassifier())])
pipeline_svm = Pipeline([('scalar3', StandardScaler()),
                      ('pca3', PCA(n_components=2)),
                      ('clf', svm.SVC())])
pipeline_knn=Pipeline([('scalar4',StandardScaler()),
                     ('pca4',PCA(n_components=2)),
                     ('knn_classifier',KNeighborsClassifier())])
pipeline_rf=Pipeline([('scalar5',StandardScaler()),
                     ('pca5',PCA(n_components=2)),
                     ('rf_classifier',RandomForestClassifier())])  
pipeline_nb=Pipeline([('scalar6',StandardScaler()),
                     ('pca6',PCA(n_components=2)), 
                     ('nb_classifier',GaussianNB())])    
pipelines = [pipeline_lr, pipeline_dt, pipeline_svm, pipeline_knn, pipeline_rf, pipeline_nb]
pipe_dict = {0: 'Logistic Regression-', 1: 'Decision Tree-', 2: 'Support Vector Machine-',3:'K Nearest Neighbor-',4:'Random Forest-',5:'Naive Bayes'}
for pipe in pipelines:
  pipe.fit(x_sm, y_sm)
for i,model in enumerate(pipelines):
    print("{} Test Accuracy : {}".format(pipe_dict[i],model.score(X_test,Y_test)))






from sklearn.model_selection import train_test_split
X_train_t, X_test_t, Y_train_t, Y_test_t = train_test_split(x_sm, y_sm, test_size=0.20, random_state=0)

 
#Random Forest Classifier:
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=50, criterion='entropy')
random_model = classifier.fit(X_train_t,Y_train_t)

y_pred = classifier.predict(X_test_t)


from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(Y_test_t,y_pred)

accuracy_score(Y_test_t,y_pred)*100

filename = 'nlp_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

'''
#Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

parameter = [{'n_estimators':[10,25,50],
              'criterion':['gini','entropy'],
              'bootstrap':[True,False],
              'max_depth':[5,6]}]

grid_search = GridSearchCV(classifier,
                           param_grid=parameter,
                           scoring='accuracy',
                           cv=10)

grid_search = grid_search.fit(X_train_t,Y_train_t)

grid_search.best_params_

classifier = RandomForestClassifier(n_estimators=100, bootstrap=False, max_depth=6, criterion='entropy')
classifier.fit(X_train_t,Y_train_t)

y_pred = classifier.predict(X_test_t)

confusion_matrix(Y_test_t,y_pred)
confusion_matrix
accuracy_score(Y_test_t,y_pred)*100
'''

