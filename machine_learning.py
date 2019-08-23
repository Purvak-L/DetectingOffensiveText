import pandas as pd
import json
import pprint
import numpy as np
import os

import string
import unicodedata
import re

import matplotlib.pyplot as plt 

from nltk.sentiment.vader import SentimentIntensityAnalyzer
#analyser = SentimentIntensityAnalyzer()

import nltk
# from nltk.tokenize.t import ToktokTokenizer
# tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')

from gensim import corpora, models
from keras import layers, models, optimizers
from sklearn.decomposition import LatentDirichletAllocation
from yellowbrick.classifier import ClassificationReport

pd.options.display.max_rows
pd.set_option('display.max_colwidth', -1)

#preprocessing pipeline
#Pipeline models features like word count, tfidf, word density, word embeddings (GloVe)

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from keras.preprocessing import text, sequence
from sklearn.metrics import f1_score, classification_report

from preprocessing import PreprocessingPipeline


class LoadDataframe:

	def __init__(self,file1, file2):
		self.file1 = file1
		self.file2 = file2
		self.df = pd.DataFrame()


	def load(self):

		hs = pd.read_csv(self.file1, encoding="ISO-8859-1",index_col=6, keep_default_na=False)
		#print(hs.head())

		orig = pd.read_csv(self.file2, index_col=0, header=None)
		orig.index.name = 'ID'
		orig = orig.rename(columns={1: 'Class'})
		orig.index = orig.index.astype(str)
		

		#merging the two dataframes
		hs = pd.merge(hs, orig, how='inner', left_index=True, right_index=True)
		self.df = hs
		self.df = self.df.dropna()
		self.df = self.df[['Tweets','Class']]
		self.df.columns = ['data','label']
		return self.df


class Feature:

	def __init__(self, df):
		self.df = df

		self.train_x, self.valid_x, self.train_y, self.valid_y = model_selection.train_test_split(df['data'], df['label'])
		
		#Label Encoder converts yes and no into 1 and 0
		encoder = preprocessing.LabelEncoder()
		
		self.train_y = encoder.fit_transform(self.train_y)
		self.valid_y = encoder.fit_transform(self.valid_y)

	def count_vectorizer(self):
		#Count vectorizer will calculate count of every word in text data and will ignore number and whitespaces

		count_vector = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
		count_vector.fit(self.df['data'])

		#using the count vector defined above, we'll transform our existing text data into train_x_count where every row will indicate 
		#tweet and every column will represent count of word indexed at that loc

		train_x_count = count_vector.transform(self.train_x)
		valid_x_count = count_vector.transform(self.valid_x)

		return train_x_count, valid_x_count

	def word_tf_idf(self):

		tfidf_vect = TfidfVectorizer(analyzer='word',token_pattern=r'\w{1,}', max_features=5000)
		tfidf_vect.fit(self.df['data'])
		xtrain_tfidf = tfidf_vect.transform(self.train_x)
		xvalid_tfidf = tfidf_vect.transform(self.valid_x)

		return xtrain_tfidf, xvalid_tfidf

	def ngram_tdidf(self):

		tfidf_ngram = TfidfVectorizer(analyzer='word',token_pattern=r'\w{1,}', ngram_range=(2,3) ,max_features=5000)
		tfidf_ngram.fit(self.df['data'])
		xtrain_tfidf_ngram = tfidf_ngram.transform(self.train_x)
		xvalid_tfidf_ngram = tfidf_ngram.transform(self.valid_x)

		return xtrain_tfidf_ngram, xvalid_tfidf_ngram



class MachineLearning:

	def __init__(self, df):

		feature_instance = Feature(df)

		self.train_x_count, self.valid_x_count = feature_instance.count_vectorizer()
		self.xtrain_tfidf, self.xvalid_tfidf = feature_instance.word_tf_idf()
		self.xtrain_tfidf_ngram, self.xvalid_tfidf_ngram = feature_instance.ngram_tdidf()

		self.train_y, self.valid_y = feature_instance.train_y, feature_instance.valid_y
		


	def train_model(self,classifier, feature_vector_train,label, feature_vector_valid, is_neural_net = False):
	    
	    classifier.fit(feature_vector_train, label)
	    prediction = classifier.predict(feature_vector_valid)

	    if is_neural_net:
	        prediction = prediction.argmax(axis=-1)
	    
	    print(classification_report(prediction, self.valid_y))

	    return metrics.accuracy_score(prediction, self.valid_y), f1_score(prediction, self.valid_y,average=None)


	def support_vector_machine(self):

		from sklearn import svm
		from sklearn.model_selection import GridSearchCV

		parameters = {'kernel':('linear', 'rbf'),'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
		
		svc = svm.SVC(gamma="scale")
		svc = GridSearchCV(svc, parameters, cv=5)

		accuracy, f1 = self.train_model(svc, self.train_x_count, self.train_y, self.valid_x_count)
		print("SVM (Count Vectors)",accuracy, f1)
		
		accuracy, f1 = self.train_model(svc, self.xtrain_tfidf, self.train_y, self.xvalid_tfidf)
		print("SVM (TF-IDF)", accuracy, f1)
		
		accuracy, f1 = self.train_model(svc, self.xtrain_tfidf_ngram, self.train_y, self.xvalid_tfidf_ngram)
		print("SVM (TDIDF-ngram)", accuracy, f1)

		print("Best parameters! -", svc.best_params_)

		return svc

	def logistic_regression(self):
		from sklearn.model_selection import GridSearchCV

		parameters = {'penalty':['l1','l2'],'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}

		clf_log = linear_model.LogisticRegression()
		clf_log = GridSearchCV(clf_log, parameters, cv=5)

		accuracy, f1 = self.train_model(clf_log, self.train_x_count, self.train_y, self.valid_x_count)
		print("Logistic Regression (Count Vectors)",accuracy, f1)
		
		accuracy, f1 = self.train_model(clf_log, self.xtrain_tfidf, self.train_y, self.xvalid_tfidf)
		print("Logistic Regression)", accuracy, f1)
		
		accuracy, f1 = self.train_model(clf_log, self.xtrain_tfidf_ngram, self.train_y, self.xvalid_tfidf_ngram)
		print("Logistic Regression (TDIDF-ngram)", accuracy, f1)

		print("Best parameters! -", clf_log.best_params_)

		return clf_log

	def naive_bayes(self):

		from sklearn.model_selection import GridSearchCV

		parameters = {'alpha':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}

		clf_log = naive_bayes.MultinomialNB()
		clf_log = GridSearchCV(clf_log, parameters, cv=5)

		accuracy, f1 = self.train_model(clf_log, self.train_x_count, self.train_y, self.valid_x_count)
		print("Logistic Regression (Count Vectors)",accuracy, f1)
		
		accuracy, f1 = self.train_model(clf_log, self.xtrain_tfidf, self.train_y, self.xvalid_tfidf)
		print("Logistic Regression)", accuracy, f1)
		
		accuracy, f1 = self.train_model(clf_log, self.xtrain_tfidf_ngram, self.train_y, self.xvalid_tfidf_ngram)
		print("Logistic Regression (TDIDF-ngram)", accuracy, f1)

		print("Best parameters! -", clf_log.best_params_)

		return clf_log





if __name__ == '__main__':

	instance_load_dataframe = LoadDataframe("hatespeech.csv","NAACL_SRW_2016.csv")
	df = instance_load_dataframe.load()
	instance_preprocessing = PreprocessingPipeline()
	df['data'] = df['data'].apply(lambda x: instance_preprocessing.normalize(x))
	machine_learning = MachineLearning(df)
	machine_learning.support_vector_machine()
	machine_learning.logistic_regression()

	print(df.head())



	
