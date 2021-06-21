"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	import nltk

	import numpy as np
	import pandas as pd

	import matplotlib.pyplot as plt
	
	import seaborn as sns

	import re
	from string import punctuation
	from nltk.tokenize import TreebankWordTokenizer
	from nltk.stem import WordNetLemmatizer
	from nltk.corpus import stopwords
	from wordcloud import WordCloud
	from collections import Counter
	from sklearn import metrics
	

	from sklearn.model_selection import train_test_split

	from sklearn.linear_model import LogisticRegression
	from sklearn.svm import SVC, LinearSVC
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.ensemble import RandomForestClassifier

	#To Look
	from sklearn.naive_bayes import MultinomialNB
	from sklearn.linear_model import SGDClassifier
	from sklearn.ensemble import AdaBoostClassifier

	# set plot style
	sns.set(style = 'whitegrid')

	

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "Exploratory Data Analysis and Insights","Model Explanation", "Aim of Our App","Team Members"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.title("Information")
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

	if selection == "Exploratory Data Analysis and Insights":
		st.title("Luke Rocks Data Analysis and Insights")
		
		st.write("""
		
		""")
		st.info('This page contains various key data insights that guided our Exploration of our data, and the factors of data preprocessing and visualisations that we utilised. ')
		from PIL import Image
		image = Image.open('Images/format_of_data.png')
		
		st.image(image, caption='')
		st.write("This is the format our data is in. We have messages, its respective tweet ID as well as the message's sentiment with regards to climate change. There are 4 sentiment expressions, namely;")
		st.write("* 1 Pro: The tweet supports the belief of our man made impact on climate change. ")
		st.write("* 2 News: the tweet links to factual news about climate change.")
		st.write("* 0 Neutral: the tweet neither supports nor refutes the belief of man-made climate change.")
		st.write("* 1 Anti: the tweet does not believe in man-made climate change.")
		from PIL import Image
		image = Image.open('Images/counts_of_class.png')
		st.image(image, caption='')

		st.write("From the figure above, we observed that we have unbalanced classes. * The majority of tweets (53.9%) support the belief of man-made climate change. * 23% consist of factual news regarding climate change. * 14.9% are neutral about man-made climate change* 8.2% don't believe in man-made climate change")

		st.write("")
		st.write("")

		st.write("  Next, lets investigate into the number of unique words used in each class.")

		from PIL import Image
		image = Image.open('Images/box_plot.png')
		st.image(image, caption='Number of words for corresponding sentiment class')

		st.write("Tweets representing news contain less words. People who believe in man-made climate change appear to used on average the same ammount of words")

		st.write("Now let's us study the distribution of the length of the words.")
		st.write("* First we obtained a list containing all the words. * Afterwards we obtained the lenth of each word and counted the number of times the word appears in our list. * Lastly we grouped frequencies by lenght and summed them up.")

		from PIL import Image
		image = Image.open('Images/word_length.png')
		st.image(image, caption='')

		st.write("The lengths of the words ranged from 1-70, to obtain a better visualisation we limited the domain to words of lengths 1-20. The length of the words appears to be positively skewed. We can expect the data to contain outliers to the right of the distribution. Most words lengths (78.4%) lies between 3-8,with the peak being 7.")
		
	if selection == "Aim of Our App":
		st.title('Title for the page')
		st.write("Here's our first attempt at using data to create a table:")
		st.write(pd.DataFrame({
    	'first column': [1, 2, 3, 4],
    	'second column': [10, 20, 30, 40]
		}))

		"""
		# My first app
		Here's our first attempt at using data to create a table:
		"""

		df = pd.DataFrame({
  		'first column': [1, 2, 3, 4],
  		'second column': [10, 20, 30, 40]
		})

		df


		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
