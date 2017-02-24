import os
import re
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor

def sample_by_year(df, yr_range):
	return df[(df["year"]>=yr_range[0]) & (df["year"]<=yr_range[1])]

if __name__ == "__main__":
	## load data ##
	subsample = False

	columns = ["id", "submission_time", "created_at_i", "author", "upvotes", "url", "num_comments", "headline"]
	submissions = pd.read_csv("~/dataHN/stories.csv", names=columns)

	## setup useful columns and remove others ##

	drop_cols = ["id", "created_at_i", "author", "url", "num_comments"]
	submissions = submissions.drop(drop_cols, axis=1).dropna()
	dates = submissions["submission_time"].str.split('-').tolist()
	submissions.drop(["submission_time"], axis=1, inplace=True)
	submissions["year"] = list(map(lambda x: int(x[0]), dates))
	print(submissions.head())

	## Optional year filtering ##
	rowo = submissions.shape[0]
	if subsample:
		year_range = (2010, 2015)
		submissions = sample_by_year(submissions, year_range)
	print("Percentage of data selected: {0}%".format((submissions.shape[0]/rowo)*100))
	#print(submissions["year"].unique())

	## Text Cleaning and Preprocessing ##

	clean_headlines = [re.sub("[^a-zA-Z0-9:!?]", " ", hl).lower() for hl in submissions["headline"]]


	## Tokenization ##

	clean_tokenized = [hl.split(" ") for hl in clean_headlines]

	#clear some memory for large bag of words array
	tokenized_headlines = None
	tokenCmb = None
	tokens = None
	tokenCounts = None
	
	## Bag of words model ##

	vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = "english", max_features = 5000)
	counts = vectorizer.fit_transform([tokens[0] for tokens in clean_tokenized])
	
	## Normalize target 0-1 ##

	submissions["upvotes_norm"] = (submissions["upvotes"] - submissions["upvotes"].min()) / (submissions["upvotes"].max() - submissions["upvotes"].min())

	## Splitting the data: 60/40##

	X_train, X_test, y_train, y_test = train_test_split(counts, submissions["upvotes_norm"], test_size=0.4, random_state=1)
	counts = None
	submissions = None

	## Predicting upvotes based on article Headlines, ##
	## using bag of words with 5000 features ##
	
	## Neural Network ##
	## Multi-layer Perceptron (supervised) ##

	start_time = time.time()
	mlp = MLPRegressor(solver='sgd', hidden_layer_sizes=(100,), activation='tanh', alpha=1e-5, learning_rate_init=1e-3, learning_rate='constant', random_state=1)

	mlp.fit(X_train, y_train)
	train_time = time.time()

	predictions = mlp.predict(X_test)

	## Mean Squared Error ##

	mse = np.mean((y_test - predictions)**2)
	print(mse)
	print("--Training Time--  {0}".format(train_time - start_time))
	
	print(mlp.n_layers_, '\n', [coef.shape for coef in mlp.coefs_], '\n', [intercept.shape for intercept in mlp.intercepts_], '\n')

	
	
