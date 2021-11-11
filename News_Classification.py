import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import re
import string
from pathlib import Path
import pickle
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# import io
# import os
# import shutil
import zipfile
# import cPickle
# import urllib
# import tensorflow as tf

# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, LSTM, Dropout, Bidirectional, Input, Conv1D, MaxPooling1D
# from tensorflow.keras.layers import TextVectorization
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

filename = 'randomforestclassifier_model.pkl'

zip_name = 'rf_clf.zip'

@st.cache(allow_output_mutation=True)
def load_model():
  with zipfile.ZipFile(zip_name, 'r') as zp:
    rf_model = pickle.loads(zp.open(filename).read())

  return rf_model

def transform_word(texts):
    texts_new = []
    for text in texts:
        text = re.sub('\w*\d\w*', '', 
                  re.sub('\n', '',
                re.sub('[%s]' % re.escape(string.punctuation), '', 
                  re.sub('<.*?>+', '', 
                        re.sub('https?://\S+|www\.\S+', '', 
                                re.sub("\\W", ' ', 
                                      re.sub('\[.*?\]', '', text.lower())))))))
        texts_new.append(text)
    return np.array(texts_new)


def load_image(filepath, width=300):
  img = Image.open(filepath)
  return st.image(img, width=width)

# @st.cache(allow_output_mutation=True)
# def load_model():
#   with open(filename, 'rb') as pkfile:
#     rf_model = pickle.load(pkfile)

#   return rf_model


if __name__ == '__main__':
  st.title('FAKE NEWS CLASSIFICATION APP BY TEAM APACHE')
  st.write('Fake news Classification using Machine Learning')
  load_image('images/fake_or_real_news.jpg', 600)
  st.subheader('Enter Your News Content below')
  sentence = st.text_area('Your News Area', 'Some news', height=200)
  predict_btn = st.button('Make Prediction')
  rf_model = load_model()

if predict_btn:
  prediction = rf_model.predict([sentence])

  pred_prob = rf_model.predict_proba([sentence])[:, 1]

  if prediction == 0:
    st.warning('This is a fake news')
    load_image('Fake_News.png')
    st.info(f'The Model predicts that there is a {100 - pred_prob[0]*100}% \
          probability that the news is fake.')

  if prediction == 1:
    st.success('This is not a fake news, It is True.')
    load_image('True_News.jpg')
    st.info(f'The Model predicts that there is a {pred_prob[0]*100}% \
          probability that the news is true.')