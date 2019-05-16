from sklearn.model_selection import train_test_split
import numpy as np
import os
import glob
from bs4 import BeautifulSoup as bs
from bs4 import BeautifulStoneSoup
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfVectorizer

class ModelHandler:
    def __init__(self):
        self.tf_vectorizer = TfidfVectorizer(min_df = 1, max_df = 0.8, ngram_range=(1,3))
        self.count_vect = CountVectorizer(min_df = 1, max_df = 0.8, ngram_range=(1,3))

    def read_file(self, file_path = "sentiment_data/dvd", file_name = "negative.review"):
#         files = glob.glob(file_path)
        data = []
        labels = {}
        sentiment = file_name.split(".")[0]
        review_file = os.path.join(file_path, file_name)
        file = glob.glob(review_file)[0]

        with open(file) as review:
            xml_string = review.read()
            soup = bs(xml_string)

            for i in soup.find_all('review_text'):
                text = bs(i.get_text())
                data.append(text.get_text())
        return data

    def create_dataframe(self, dir_name):
        final_data = []
        for filename in ['positive.review', 'negative.review']:
            data = self.read_file(file_path=dir_name, file_name=filename)
            sentiment = filename.split(".")[0]
            label = 1 if sentiment == "positive" else 0
            final_data.append({"review" : data, "sentiment" : label })

        df = pd.DataFrame().from_dict(final_data[0])
        df2 = pd.DataFrame().from_dict(final_data[1])
        df = df.append(df2)
        return df

    def vectorize_dataset(self, dir_name, select_vectorizer):        
#         Create a dataframe with reviews and its sentiment
        if(select_vectorizer == "count"):
            self.vectorizer = self.count_vect
        else:
            self.vectorizer= self.tf_vectorizer
        
        df = self.create_dataframe(dir_name)
        
        X= df.loc[:, 'review']
        Y= df.loc[:, 'sentiment']

#         Split the data into training set and test set for validation
        x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
        tf = self.vectorizer.fit_transform(X)

        tf_columns = list(self.vectorizer.fit(X).vocabulary_.keys())
        
        df31 = pd.DataFrame(tf.toarray(), columns=tf_columns)
        
        df31['sentiment'] = Y.tolist()
        self.XT_train, self.XT_test, self.YT_train, self.YT_test = train_test_split(df31.iloc[:, :-1], df31.iloc[:, -1], test_size = 0.2, random_state = 0)
        
    def create_multinomial_classifier(self, classifier):
        classifier.fit(self.XT_train, self.YT_train)

    def test_multinomial_classifier(self, classifier):
        test = classifier.predict(self.XT_test)
        return test
    
    def get_vectorizer(self):
        return self.vectorizer

    def predict(self, sentence, classifier):
        pred = classifier.predict(sentence)
        return pred