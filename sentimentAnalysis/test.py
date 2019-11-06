import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.image as img
import numpy as np
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pyrebase
from django.contrib import auth
from os import path
from PIL import Image

import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk import FreqDist
from wordcloud import WordCloud

import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score


from pycorenlp import StanfordCoreNLP

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
from sklearn.metrics import cohen_kappa_score
import numpy as np

#	annotation 전체 개수=> form.wordcounter
#	positive 전체 개수 => form.positiveWordcounter
#	negative 전체 개수 => form.negativeWordcounter

def word_graph(wordconter, positiveWordcounter, negativeWordcounter):
    pos_per = (positiveWordcounter/wordconter)*100
    neg_per = (negativeWordcounter/wordconter)*100
    neu_per = 100-pos_per-neg_per
    A = [pos_per]
    B = [neu_per]
    C = [neg_per]
    a_b= [C[0]+B[0]]

    plt.rcParams['figure.figsize'] = [2, 10]
    plt.bar(1, C, color="red" )
    plt.bar(1, B, bottom = C, color="orange")
    plt.bar(1, A, bottom = a_b, color = "green")
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("fig.png", format='png')


label = ["확인"] 
A = [20] #positive
B = [10] #netural
C = [70] #negative

c_b = [A[0]+B[0]]
a_b = [C[0]+B[0]]

X =1
plt.rcParams["figure.figsize"] = (4,14)
plt.bar(1, C, color="red" )
plt.bar(1, B, bottom = C, color="orange")
plt.bar(1, A, bottom = a_b, color = "green")
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
plt.savefig("fig.png", dpi=300)

"""
plt.figure(figsize=(20, 10))
img = mpimg.imread('fig1.png')
imgs = img.transpose(1, 0, 2)
plt.imshow(imgs)
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.savefig('fig1.png')
plt.show()

"""

'''
def vaderSentimentFucntion(sentences):
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    polarities = []
    count_pol = []
    for sentence in sentences:
        positive = 0
        negative = 0
        neutral = 0
        vs = analyzer.polarity_scores(sentence)
        scores.append(vs['compound']) #only Compound value
        if vs['compound'] >= 0.05:
            polarities.append("Positive")
            positive += 1
            count_pol.append([positive, neutral, negative])
        elif vs['compound'] <0.05 and vs['compound'] >-0.05:
            polarities.append("Neutral")
            neutral += 1
            count_pol.append([positive, neutral, negative])
        elif vs['compound']<=-0.05:
            polarities.append("Negative")
            negative += 1
            count_pol.append([positive, neutral, negative])
    return scores, polarities, count_pol



def preprocessFile(text):
    id = []
    content = []
    hashtag = []
    annotation = []

    pattern = '#([0-9a-zA-Z]*)'
    hashtag_word = re.compile(pattern)


    sentence = re.split(r'\t+', text)
    text = ""
    id.append(sentence[0])
    content.append(sentence[1])

    for tag in hashtag_word.findall(text):
        hashtag.append(tag)

    return id,content,annotation, hashtag

def sentenceLevel(text):
    result = []
    for tweet in text:
        result.append(nltk.sent_tokenize(tweet))
    return result


def save_wordcloud(text,fileName):
    wc = WordCloud(width=1000, height=600, background_color="white", random_state=0)
    wc.generate_from_frequencies(text)
    wc.to_file(path.join("sentimentAnalysis/static/img/", fileName+".png"))


def word_frequent(text):
    fd_content = FreqDist(text)
    return fd_content

def cleansing(content):
    result = []
    url_pattern ='https?://\S+|#([0-9a-zA-Z]*)'

    for text in content:
        #대문자 소문자 변환
        lower_content = (text.lower())

        #불용어 제거
        shortword = re.compile(r'\W*\b\w{1,2}\b')
        shortword_content = shortword.sub('', lower_content)
        text = re.sub('[-=.#/?:$}!,@]', '', shortword_content)

        stop_words = set(stopwords.words('english'))
        content_tokens = word_tokenize(text)
        real_content = re.sub(pattern=url_pattern, repl='', string = text)

        for w in content_tokens:
            if w not in stop_words:
                    result.append(w)
    return result

def vaderSentimentFucntion_sentence(sentences):
    scores = []
    polarities = []
    count_polarity = []
    count = len(sentences)
    for i in range(count):
        score, polarity, count_pol = vaderSentimentFucntion(sentences[i])
        scores.append(score)
        polarities.append(polarity)
        count_polarity.append(count_pol)
    return scores, polarities, count_polarity

tweet = ['tweet1_sentence1. tweet1_sentence2', 'tweet2_sentence1. tweet2_sentence2'] 
sentence = [['tweet1_sentence1.', 'tweet1_sentence2'], ['tweet2_sentence1.', 'tweet2_sentence2']]

text = "#AntiFascism I love you. I need you. Disea... http://t.co/Q6xg2hmh6i #Anarchy	Negative"
['#AntiFascism I love you. I need you. Disea... http://t.co/Q6xg2hmh6i #Anarchy']
[['#AntiFascism I love you.', 'I need you.', 'Disea... http://t.co/Q6xg2hmh6i #Anarchy']]
id,content,annotation, hashtag = preprocessFile(text)
result = sentenceLevel(tweet)
scores, polarities, count_polarity = vaderSentimentFucntion_sentence(sentence)
print(count_polarity)


#texts =["#AntiFascism Documentary Aired on Danish Television Exposes HPV Vaccines for Triggering Wave  coffee phone phone phone cat dog doh of Disea... http://t.co/Q6xg2hmh6i #Anarchy"]
#text = "coffee phone phone phone cat dog doh" 
#result = cleansing(texts)
#print(result)
#fd_content = word_frequent(result)
a ="abc"
#print(fd_content)
#wc = WordCloud(width=1000, height=600, background_color="white", random_state=0).generate(text)
#wc.to_file(path.join("sentimentAnalysis/static/img/", "text.png"))
'''
