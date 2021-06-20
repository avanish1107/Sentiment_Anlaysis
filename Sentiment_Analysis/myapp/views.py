from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
import numpy as np

# Create your views here.
def index(request):
	return render(request,'index.html')
def submit(request):

	# reading the data set 
	df = pd.read_csv("/home/user/GroupProject/amazon.txt",sep='\t',names=['txt','liked'])

	# Word tokenizer  and removal of stop words
	stopset=set(stopwords.words('english')) 
	 # transformation from upper case to lower case 
	vectorizer = TfidfVectorizer(use_idf=True,lowercase=True,strip_accents='ascii',stop_words=stopset)

	#  Detetminig independent variable 
	y=df.liked
	# Determining the dependent variable 
	x=vectorizer.fit_transform(df.txt)


	# Training and testing the data with some random set of data 
	''' Random state is a seed used by the random number generator 
		If random state instance , random state is the random number generator;
		If 'none' the random number generator is the random state instance used by np.random. '''
	# to maximize the accuracy we have taken random number generator as 42
	x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)  
	# Applying the Naive Bayes Multinomial Algorithm
	clf=naive_bayes.MultinomialNB()
	#Training the data by applying Naive Bayes Algorithm
	clf.fit(x_train,y_train)

	# Calculation of accuracy of the data 
	accuracy=roc_auc_score(y_test,clf.predict_proba(x_test)[:,1])
	accuracy=accuracy*100
	accuracy=str(accuracy)
	
	# reading the data from the user through webpage. 
	a=request.GET['review']
	reviews_arr=np.array([a])
	reviews_vector=vectorizer.transform(reviews_arr) # Applying the  Sentiment analysis on the input data .
	# Predicting the sentiment according to the given input 
	b = clf.predict(reviews_vector)
	if b==1:
	    response="Positive Review! \nThe accuracy of the result is "
	if b==0:
	    response="Negative Review! \nThe accuracy of the result is  "
	response=response+accuracy
	#Returning the response to the webpage 
	return HttpResponse(response)


