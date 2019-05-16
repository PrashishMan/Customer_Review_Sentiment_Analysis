import nltk
import glob
import os
import pandas as pd
import xml.etree.ElementTree as ET
import re
from bs4 import BeautifulSoup as bs
from bs4 import BeautifulStoneSoup
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk import bigrams, FreqDist

# Extraciting review from the files
class NaiveBayesAlgorithm():
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

    def get_combined_text(self,series):
        combined_text = ""
        for i in series.iteritems():
            combined_text += i[1]
        return combined_text    
        
    def tokenizer(self,sentence):
    #   Converting sentences into lower characters and removing any punctuations
        normalize_words = self.normalize(sentence)    
    #   Converting sentences into list of words
        
        tokenize_words = self.tokenize(normalize_words)
        
    #   converting sentences into base forms
        stemm_lemma_words = self.stem_lemma_word(tokenize_words)
        
        return stemm_lemma_words
    
    def get_vocabulary(self,series):
        sentences = self.get_combined_text(series)
        words = self.tokenizer(sentence=sentences)
        bigrams_words = bigrams(words)
        bigrams_freq = FreqDist(bigrams_words)
        double_words = []
        
        for key, value in bigrams_freq.items():
            double_words.append(key[0] + " " + key[1])
        
        for word in double_words:
            words.append(word)
        
        return set(words)
        
    def get_BOW(self, series):
        words = self.get_vocabulary(series)
        vect = CountVectorizer(vocabulary=words, min_df=2, max_df=0.8)
        vect2 = vect.fit_transform(series)
        vect_list = vect2.toarray()
        cv_columns = list(vect.fit(series).vocabulary_.keys())
        
        df_count = pd.DataFrame(vect_list, columns=cv_columns)
        return df_count
    
    def normalize(self,review):
        review = review
        review = review.lower()
        #remove punctuation(.,!,:)
        review = re.sub(r"[^a-zA-Z0-9]", " ", review)
        
        return review

    def tokenize(self, words):
        words = np.array(word_tokenize(words))
        unique_words = set()
        for i in words:
            unique_words.add(i)
        return unique_words

    # stemming and lemmatization
    def stem_lemma_word(self,words):
        valid_words = []
        stemmed = [PorterStemmer().stem(w) for w in words]
        lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed]
        for word in lemmed:
            if word not in stopwords.words('english'):
                valid_words.append(word)
        return valid_words
    
    
    
    # Sentiment Classification algorithm
    # Preprocessing new sentecnes
    def pred_text_preprocessing(self,text):
        words = self.normalize(text)
        words = word_tokenize(text)
        words = self.stem_lemma_word(words)
        return words

    # Creating a model
    # Calculating the probablity
    def calculate_prob(self,sentence, df2, total_word_positive_count, total_word_negative_count, positive_prob, negative_prob, vocabulary):
    #     Calculating the prior probablity
        words_count = dict()
        
        prop_yes_given_word = 1
        prop_no_given_word = 1
        
    #   Calculating conditional probablity 
    #   Calculating the positive probablity
        for word in self.pred_text_preprocessing(sentence):
            if(word and word in df2.columns.tolist()):
                prop_yes_given_word *= (np.sum(df2[(df2["sentiment"] == 1)][word]) + 1)/\
                (total_word_positive_count + vocabulary)
                prop_no_given_word *= (np.sum(df2[(df2["sentiment"] == 0)][word]) + 1)/\
                (total_word_negative_count + vocabulary)
        
        prop_yes_given_word = positive_prob * prop_yes_given_word    
        prop_no_given_word = negative_prob * prop_no_given_word
            
        if(prop_yes_given_word > prop_no_given_word):
            return "Positive"
        
        return "Negative"

    #making predictions
    def make_predictions(self,sentence, probablity_df):
        total = len(probablity_df["sentiment"])
        positive = len(probablity_df[(probablity_df["sentiment"] == 1)])
        negative = total-positive

        vocabulary = len(probablity_df.columns)
        tp_count = np.sum(np.sum(probablity_df[probablity_df['sentiment'] == 1]))
        tn_count = np.sum(np.sum(probablity_df[probablity_df['sentiment'] == 0]))

        #   Calculating the prior probablities
        positive_prob = positive/total
        negative_prob = negative/total
        
        prob = self.calculate_prob(sentence, probablity_df, total_word_positive_count = tp_count,
                            total_word_negative_count = tn_count, positive_prob = positive_prob,
                            negative_prob = negative_prob, vocabulary = vocabulary)
        
        return prob  
