from CustomModel import NaiveBayesAlgorithm
from ModelHandler import ModelHandler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn import svm

algo = NaiveBayesAlgorithm()
def initialize_custom_parameters():
    df = algo.create_dataframe("sentiment_data/electronics")
    X= df.loc[:, 'review']
    Y= df.loc[:, 'sentiment']

    x_train, x_test,y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

    probablity_df = algo.get_BOW(x_train)
    probablity_df['sentiment'] = y_train.tolist()

    total = len(probablity_df["sentiment"])
    positive = len(probablity_df[(probablity_df["sentiment"] == 1)])
    negative = total-positive

    vocabulary = len(probablity_df.columns)
    print("3 calculating total positive count ")
    tp_count = np.sum(np.sum(probablity_df[probablity_df['sentiment'] == 1]))
    print("2 calculating total negative count ")
    tn_count = np.sum(np.sum(probablity_df[probablity_df['sentiment'] == 0]))
    print("1 Preparing the classifier ... ")
    #   Calculating the prior probablities
    positive_prob = positive/total
    negative_prob = negative/total

    return probablity_df, tp_count, tn_count, positive_prob, negative_prob, vocabulary

def calculate_custom_nb_prob():
    next = 'Y'
    probablity_df, tp_count, tn_count, positive_prob, negative_prob, vocabulary = initialize_custom_parameters()
    while(next == 'Y'):
        
        print("This is a sentiment analysis application for product review")
        review= input("Please enter your review : ")
        print("Determining the sentiment .... ")
        prob = algo.calculate_prob(review, probablity_df, total_word_positive_count = tp_count,
                                    total_word_negative_count = tn_count, positive_prob = positive_prob,
                                    negative_prob = negative_prob, vocabulary = vocabulary)
                                
        print(prob)
        next= input("Do you want to enter next review? (Y/N) : ")
        while next not in ['Y', 'N']:
            print("Please enter Y (yes) or N (no)!! ")
            next= input("Do you want to enter next review? (Y/N) : ")

def calculate_nb_prob(clas, classifier):
    next = 'Y'
    while(next == 'Y'):
        print("This is a sentiment analysis application for product review")
        review= input("Please enter your review : ")
        print("Determining the sentiment .... ")
        vectorizer = clas.get_vectorizer()
        test_review = [review,]
        vectorized_review = vectorizer.transform(test_review)

        pred = clas.predict(vectorized_review, classifier)
        sentiment = "Positive" if pred[0] == 1 else "Negative"   
        print(sentiment)
        next= input("Do you want to enter next review? (Y/N) : ")
        while next not in ['Y', 'N']:
            print("Please enter Y (yes) or N (no)!! ")
            next= input("Do you want to enter next review? (Y/N) : ")

def get_vectorizer():
    print("Please select the vectorizer ")
    print("1: Count Vectorizer")
    print("2: TFIDF Vectorizer")

    vectorizer_ind = input("Vectorizer : ")
    try:
        vectorizer_ind = int(vectorizer_ind)
    except ValueError:
        pass

    print(vectorizer_ind)
    while(vectorizer_ind not in [1,2]):
        print("Error !! Invalid number selected")
        vectorizer_ind = input("Vectorizer : ")
        try:
            vectorizer_ind = int(vectorizer_ind)
        except ValueError:
            pass

    vectorizer_name = "count" if vectorizer_ind == 1 else "tfidf"
    print(vectorizer_name)

    return vectorizer_name

if __name__ == "__main__":
    print("Please select the number for classifier type : ")
    print("1: Custom Naive Bayes Classifier ")
    print("2: Naive Bayes Classifier ")
    print("3: Decision Tree Classifier ")
    print("4: Support Vector Machines ")
    print("5: exit ")

    index = input("Please select the number : ")
    try:
        index = int(index)
    except ValueError:
        pass

    while(index not in [1,2,3,4,5]):
        index = input("Please select the number : ")
        try:
            index = int(index)
        except ValueError:
            pass

    if int(index) == 1:
        print("4 selectd")
        calculate_custom_nb_prob()
    
    # Using naive bayes model
    elif int(index) == 2:
        vectorizer_name = get_vectorizer()
        nb = ModelHandler()
        multinomial_nb = MultinomialNB()
        dir_name = "sentiment_data/electronics"
        nb.vectorize_dataset(dir_name, vectorizer_name)
        nb.create_multinomial_classifier(multinomial_nb)
        
        calculate_nb_prob(nb, multinomial_nb)


    # Using decision tree classifier
    elif int(index) == 3:
        vectorizer_name = get_vectorizer()

        nb = ModelHandler()
        tree_classifier = tree.DecisionTreeClassifier()
        dir_name = "sentiment_data/electronics"
        nb.vectorize_dataset(dir_name, vectorizer_name)
        nb.create_multinomial_classifier(tree_classifier)
        
        calculate_nb_prob(nb, tree_classifier)

    elif int(index) == 4:
        vectorizer_name = get_vectorizer()

        nb = ModelHandler()
        classifier_liblinear = svm.LinearSVC()
        dir_name = "sentiment_data/electronics"
        nb.vectorize_dataset(dir_name, vectorizer_name)
        nb.create_multinomial_classifier(classifier_liblinear)
        
        calculate_nb_prob(nb, classifier_liblinear)

    
    
    
     