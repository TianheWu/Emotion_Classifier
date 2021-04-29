from nltk import data
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import nltk
import numpy
import re
import string
from pandas import read_csv, DataFrame
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# return the train_text and train_label
def load_data():
    f = open(r'IMDB Dataset.csv', encoding='UTF-8')
    data = read_csv(f, names=numpy.arange(2))
    data_df = DataFrame(data)
    text_data = []
    label_data = []
    content_text = data_df.iloc[1:49999, 0]
    label_text = data_df.iloc[1:49999, 1]
    for text in content_text:
        text_data.append(text)
    for label in label_text:
        if label == 'positive':
            label_data.append(1)
        else:
            label_data.append(0)
    return numpy.array(text_data), numpy.array(label_data)


# return text_data removed punctuation and ......
def text_process(text_data):
    processed_text_data = []
    for text in text_data:
        text = re.sub("<br /><br />", " ", text)
        text = text.lower()
        text = re.sub("\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation.replace("'", "")))
        text = text.strip()
        processed_text_data.append(text)
    return numpy.array(processed_text_data)


# return the text without stop_word and the word include '
def delete_stop_stem_word(datalist, stop_words):
    data_list_pre = []
    for text in datalist:
        text_words = word_tokenize(text)
        text_words = [word for word in text_words if not word in stop_words]
        # delete the word with '
        text_words = [word for word in text_words if len(re.findall("'", word)) == 0]
        data_list_pre.append(text_words)
    return numpy.array(data_list_pre)

