import pandas as pd
import sys
from nltk.corpus import stopwords
import pickle
import re
import numpy as np
from azureml import Workspace
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB


def azureml_main(dataframe1 = None, dataframe2 = None):
    ws = Workspace(workspace_id='28af1aad6dbc4b29aafbd9777057d089',
    authorization_token='oqFY798t13rZ1bAszi7dbAsQJ3QvC+VDyXMvCoI+ZJ9Dk4e273mDkfQvn+5yFR/35ptf7t0zvHW7xdqA7PrpsQ==',
    endpoint='https://europewest.studioapi.azureml.net')
    ds = ws.datasets['Input_data.csv']
    frame = ds.to_dataframe()
    X = frame.loc[:,'Input_text']
    Y = frame.loc[:,'Category']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, random_state=1, stratify=frame['Category'])
    count_vectorizer = CountVectorizer(stop_words='english')
    count_train = count_vectorizer.fit_transform(X_train)
    count_test = count_vectorizer.transform(X_test).toarray()
    nb = MultinomialNB()
    nb.fit(count_train, Y_train)
    nb_pred_train = nb.predict(count_train)
    nb_pred_test = nb.predict(count_test)
    
    #sys.path.insert(0,".\\Script Bundle\\azureml")
    #model = pickle.load(open(".\\Script Bundle\\azureml\\nb.pkl", 'rb'))
    #pred = model.predict(dataframe1)
    #accuracy = model.score(dataframe1, Y_test)
    
    
    #return accuracy
    return pd.DataFrame([nb_pred_test[0]]),
    #return pred[0],
#azureml_main()
    

