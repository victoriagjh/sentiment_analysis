from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import os
from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponseRedirect,HttpResponse,Http404
from .forms import UploadFileForm
from .models import SAResult,SAResultManager
from django.contrib import messages

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

from django.core.files import File
from pycorenlp import StanfordCoreNLP

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
from sklearn.metrics import cohen_kappa_score
import numpy as np


from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
# Create your views here.
session ={}
def sentimentAnalysis(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            tools=[]
            if type(request.POST.get('vader')) !=type(None):
                tools.append("Vader Sentiment")
            if type(request.POST.get('textblob')) != type(None):
                tools.append("TextBlob")
            if type(request.POST.get('sentiwordnet')) != type(None):
                tools.append("sentiWordnet")
            if type(request.POST.get('stanford')) != type(None):
                tools.append("Stanford NLP")
            if tools:
                #preprocess the file
                form.save()
                form.name = request.FILES['file'].name
                ids, contents, annotations, hashtags = preprocessFile(request.FILES['file'])

                form.ids=ids
                form.annotations=annotations
                form.content = contents #by tweet
                form.content_sentence = sentenceLevel(form.content) #by sentence
                form.hashtag = hashtags #text type(list)
                form.tool = tools

                #surface metrics
                cleansingText = cleansing(contents)
                form.word_frequent = word_frequent(cleansingText)
                form.topFrequentWords=top_freqeunt(cleansingText)
                form.wordcounter = wordcounter(cleansingText)
                save_wordcloud(form.word_frequent,"basic")
                form.hashtag_frequent = top_freqeunt(hashtags)

                #sentimentAnalysis
                for i in tools:
                    if i == "Vader Sentiment":
                        form.vaderScores, form.vaderPolarity, form.vaderCountpol =vaderSentimentFucntion(form.content) #result of tweet
                        form.vaderScores_sentence, form.vaderPolarity_sentence, form.vaderCountpol_sentence = vaderSentimentFucntion_sentence(form.content_sentence)
                        form.vaderAverage = average(form.vaderScores_sentence)
                        form.vaderMajority = majority(form.vaderPolarity_sentence)
                        form.vaderConfusionMatrix = confusionMatrix(form.annotations, form.vaderPolarity)
                        form.vaderPrecise = round(precise(form.annotations, form.vaderPolarity),2)
                        form.vaderRecall = round(recall(form.annotations, form.vaderPolarity),2)
                        form.vaderF1Score = round(F1Score(form.vaderPrecise,form.vaderRecall),2)
                    elif i == "TextBlob":
                        form.textblobScores, form.textblobPolarity, form.textblobCountpol = textblobSentimentFunction(form.content) #result of tweet
                        form.textblobScores_sentence, form.textblobPolarity_sentence, form.textblobCountpol_sentence = textblobSentimentFunction_sentence(form.content_sentence)
                        form.textblobAverage = average(form.textblobScores_sentence)
                        form.textblobMajority = majority(form.textblobPolarity_sentence)
                        form.textblobConfusionMatrix = confusionMatrix(form.annotations, form.textblobPolarity)
                        form.textblobPrecise = round(precise(form.annotations, form.textblobPolarity),2)
                        form.textblobRecall = round(recall(form.annotations, form.textblobPolarity),2)
                        form.textblobF1Score = round(F1Score(form.textblobPrecise,form.textblobRecall),2)
                    elif i == "sentiWordnet":
                        form.sentiWordnetScore, form.sentiWordnetPolarity, form.sentiWordnetCountpol = sentiWordnetSentimentFunction(form.content) #result of tweet
                        form.sentiWordnetScore_sentence, form.sentiWordnetPolarity_sentence, form.sentiWordnetCountpol_sentence = sentiWordnetSentimentFunction_sentence(form.content_sentence)
                        form.sentiWordnetAverage = average(form.sentiWordnetScore_sentence)
                        form.sentiWordnetMajority = majority(form.sentiWordnetPolarity_sentence)
                        form.sentiWordnetConfusionMatrix = confusionMatrix(form.annotations, form.sentiWordnetPolarity)
                        form.sentiWordnetPrecise = round(precise(form.annotations, form.sentiWordnetPolarity),2)
                        form.sentiWordnetRecall = round(recall(form.annotations, form.sentiWordnetPolarity),2)
                        form.sentiWordnetF1Score = round(F1Score(form.textblobPrecise,form.sentiWordnetRecall),2)
                    elif i == "Stanford NLP":
                        form.stanfordNLPPolarity, form.stanfordNLPCountpol = stanfordNLPSentimentFunction(form.content) #result of tweet
                        form.stanfordNLPPolarity_sentence, form.stanfordNLPCountpol_sentence = stanfordNLPSentimentFunction_sentence(form.content_sentence)
                        form.stanfordNLPMajority = majority(form.stanfordNLPPolarity_sentence)
                        form.stanfordNLPConfusionMatrix = confusionMatrix(form.annotations, form.stanfordNLPPolarity)

                #surface metric according to positive and negative
                positiveSet,negativeSet=separatePN(form.annotations,form.content)
                positiveList = list(positiveSet)
                negativeList = list(negativeSet)
                positiveHashtag= extractHashtag(positiveList)
                negativeHashtag= extractHashtag(negativeList)
                form.positiveTopFrequentHashtag=top_freqeunt(positiveHashtag)
                form.negativeTopFrequentHashtag=top_freqeunt(negativeHashtag)

                #surface metrics
                positiveCleansingText = cleansing(positiveList)
                form.positiveWord_frequent = word_frequent(positiveCleansingText)
                form.positiveTopFrequentWords=top_freqeunt(positiveCleansingText)
                form.positiveWordcounter = wordcounter(positiveCleansingText)
                save_wordcloud(form.positiveWord_frequent,"positive")

                negativeCleansingText = cleansing(negativeList)
                form.negativeWord_frequent = word_frequent(negativeCleansingText)
                form.negativeTopFrequentWords=top_freqeunt(negativeCleansingText)
                form.negativeWordcounter = wordcounter(negativeCleansingText)
                save_wordcloud(form.negativeWord_frequent,"negative")

                #sorted Sentiment Result F1 Score
                form.sortedF1ScoreLists = F1ScoreSorted(form.vaderF1Score,form.textblobF1Score,form.sentiWordnetF1Score)

                #for kappa calculate
                form.sumPolarity_sentence = sum_for_kappa_sentence(form.vaderCountpol_sentence, form.textblobCountpol_sentence, form.sentiWordnetCountpol_sentence, form.stanfordNLPCountpol_sentence)
                form.sumPolarity_tweet = sum_for_kappa_tweet(form.vaderCountpol, form.textblobCountpol, form.sentiWordnetCountpol, form.stanfordNLPCountpol)
                form.KappaScore_sentence = fleiss_kappa(form.sumPolarity_sentence)
                form.KappaScore_tweet = fleiss_kappa(form.sumPolarity_tweet)

                SAResultList=[]
                SAResultObjectList=[]
                for i in range(0,len(ids)):
                    temp=SAResult.objects.create_result(form.ids[i],form.content[i],form.vaderScores[i],form.vaderPolarity[i],form.textblobScores[i],form.textblobPolarity[i],
                    form.stanfordNLPPolarity[i],form.sentiWordnetScore[i],form.sentiWordnetPolarity[i],form.KappaScore_tweet[i])
                    SAResultObjectList.append(temp)

                for i in range(0,len(SAResultObjectList)):
                    SAResultList.append(SAResult.objects.filter(ids=SAResultObjectList[i].ids).values()[0])

                form.SAResultList=SAResultList

                page = request.POST.get('page',1)
                paginator = Paginator(form.SAResultList, 25)
                try:
                    pageOfTweet = paginator.page(page)
                except PageNotAnInteger:
                    pageOfTweet = paginator.page(1)
                except EmptyPage:
                    pageOfTweet = paginator.page(paginator.num_pages)

                form.pageOfTweet=pageOfTweet
                form.page=page

                context = {
                    'form':form,
                    }
                global session
                session=context

                return render(request, "expert_page.html",session)
            if not tools:
                messages.warning(request, 'You should check the tool at least 1!', extra_tags='alert')
    else:
        form = UploadFileForm()
    context = {
        'form':form,
    }
    return render(request, 'main_page.html', context)

def expert_page(request):
    if request.method == 'POST':
        global session
        if 'download' in request.POST:
            makeFile("expert")
            path_to_file = os.path.realpath("result.txt")
            file = open(path_to_file, 'r')
            rfile = File(file)
            response = HttpResponse(rfile, content_type='application/txt')
            response['Content-Disposition'] = 'attachment; filename=' + "result.txt"
            return response
    page = request.POST.get('page',1)
    paginator = Paginator(session['form'].SAResultList, 25)

    try:
        pageOfTweet = paginator.page(page)
    except PageNotAnInteger:
        pageOfTweet = paginator.page(1)
    except EmptyPage:
        pageOfTweet = paginator.page(paginator.num_pages)

    session.pageOfTweet=pageOfTweet
    session.page=page

    #tweet
    path_to_file = os.path.realpath("wpqkf.txt")
    fi=open(path_to_file,'w')
    fi.write(page)
    fi.close()

    return render(request,'expert_page.html',session)


def makeFile(pageType):
    file = open("result.txt", 'w')
    file.write("id\tcontent\tannotation\tvaderScore\tvaderPolarity\ttextblobScores\ttextblobPolarity\tstanfordNLPPolarity\n")
    for i in range(0,len(session['form'].ids)):
        file.write(session['form'].ids[i])
        file.write("\t")
        file.write(session['form'].content[i])
        file.write("\t")
        file.write(session['form'].annotations[i])
        file.write("\t")
        file.write(str(session['form'].vaderScores[i]))
        file.write("\t")
        file.write(session['form'].vaderPolarity[i])
        file.write("\t")
        file.write(str(session['form'].textblobScores[i]))
        file.write("\t")
        file.write(session['form'].textblobPolarity[i])
        file.write("\t")
        file.write(session['form'].stanfordNLPPolarity[i])
        file.write("\t")
        file.write(str(session['form'].sentiWordnetScore[i]))
        file.write("\t")
        file.write(session['form'].sentiWordnetPolarity[i])
        file.write("\n")

    file.write("Word Counter : ")
    file.write(str(session['form'].wordcounter))
    file.write("\n")
    file.write("Vader Confusion Matrix : ")
    file.write(str(session['form'].vaderConfusionMatrix[0][0]))
    file.write("\t")
    file.write(str(session['form'].vaderConfusionMatrix[0][1]))
    file.write("\t")
    file.write(str(session['form'].vaderConfusionMatrix[1][0]))
    file.write("\t")
    file.write(str(session['form'].vaderConfusionMatrix[1][1]))
    file.write("\n")
    file.write("Vader Precise : ")
    file.write(str(session['form'].vaderPrecise))
    file.write("\n")
    file.write("Vader Recall : ")
    file.write(str(session['form'].vaderRecall))
    file.write("\n")
    file.write("Vader F1 Score : ")
    file.write(str(session['form'].vaderF1Score))
    file.write("\n")

    file.write("TextBlob Confusion Matrix\n")
    file.write(str(session['form'].textblobConfusionMatrix[0][0]))
    file.write("\t")
    file.write(str(session['form'].textblobConfusionMatrix[0][1]))
    file.write("\t")
    file.write(str(session['form'].textblobConfusionMatrix[1][0]))
    file.write("\t")
    file.write(str(session['form'].textblobConfusionMatrix[1][1]))
    file.write("\n")
    file.write("TextBlob Precise : ")
    file.write(str(session['form'].textblobPrecise))
    file.write("\n")
    file.write("TextBlob Recall : ")
    file.write(str(session['form'].textblobRecall))
    file.write("\n")
    file.write("TextBlob F1 Score : ")
    file.write(str(session['form'].textblobF1Score))
    file.write("\n")

    file.write("SentiWordNet Dictionary Confusion Matrix\n")
    file.write(str(session['form'].sentiWordnetConfusionMatrix[0][0]))
    file.write("\t")
    file.write(str(session['form'].sentiWordnetConfusionMatrix[0][1]))
    file.write("\t")
    file.write(str(session['form'].sentiWordnetConfusionMatrix[1][0]))
    file.write("\t")
    file.write(str(session['form'].sentiWordnetConfusionMatrix[1][1]))
    file.write("\n")
    file.write("SentiWordNet Dictionary Precise : ")
    file.write(str(session['form'].sentiWordnetPrecise))
    file.write("\n")
    file.write("TextBlob Recall : ")
    file.write(str(session['form'].sentiWordnetRecall))
    file.write("\n")
    file.write("TextBlob F1 Score : ")
    file.write(str(session['form'].sentiWordnetF1Score))
    file.write("\n")

    file.write("Stanford NLP Confusion Matrix\n")
    file.write(str(session['form'].stanfordNLPConfusionMatrix[0][0]))
    file.write("\t")
    file.write(str(session['form'].stanfordNLPConfusionMatrix[0][1]))
    file.write("\t")
    file.write(str(session['form'].stanfordNLPConfusionMatrix[1][0]))
    file.write("\t")
    file.write(str(session['form'].stanfordNLPConfusionMatrix[1][1]))
    file.write("\n")

    file.write("Top Frequent Words 5 : ")
    for i in range(0,5):
        file.write(str(session['form'].topFrequentWords[i]))
        file.write("\t")
    file.write("\n")
    file.write("Top Frequent Hashtags 5 : ")
    for i in range(0,5):
        file.write(str(session['form'].hashtag_frequent[i]))
        file.write("\t")
    file.write("\n")
    if pageType=="expert":
        file.write("Positive Word Counter : ")
        file.write(str(session['form'].positiveWordcounter))
        file.write("\n")
        file.write("Positive Top Frequent Words 5 : ")
        for i in range(0,5):
            file.write(str(session['form'].positiveTopFrequentWords[i]))
            file.write("\t")
        file.write("\n")
        file.write("Positive Top Frequent Hashtags 5 : ")
        for i in range(0,5):
            file.write(str(session['form'].positiveTopFrequentHashtag[i]))
            file.write("\t")
        file.write("\n")
        file.write("Negative Word Counter : ")
        file.write(str(session['form'].negativeWordcounter))
        file.write("\n")
        file.write("Negative Top Frequent Words 5 : ")
        for i in range(0,5):
            file.write(str(session['form'].negativeTopFrequentWords[i]))
            file.write("\t")
        file.write("\n")
        file.write("Negative Top Frequent Hashtags 5 : ")
        for i in range(0,5):
            file.write(str(session['form'].negativeTopFrequentHashtag[i]))
            file.write("\t")
        file.write("\n")
    file.close()

def preprocessFile(f):
    fileName="text/"+f.name
    file = open(fileName, "r", encoding='UTF-8')
    text=file.readlines()

    id = []
    content = []
    hashtag = []
    annotation = []

    pattern = '#([0-9a-zA-Z]*)'
    hashtag_word = re.compile(pattern)

    for line in text:
        sentence = re.split(r'\t+', line)
        text = ""
        id.append(sentence[0])
        content.append(sentence[1])
        annotation.append(sentence[2].strip('\n'))

        for tag in hashtag_word.findall(line):
            hashtag.append(tag)

    return id,content,annotation, hashtag

def extractHashtag(list):
    hashtag=[]
    pattern = '#([0-9a-zA-Z]*)'
    hashtag_word = re.compile(pattern)
    for i in list:
        for j in hashtag_word.findall(i):
            hashtag.append(j)
    return hashtag

def separatePN(annotations,text):
    positiveSet = set()
    negativeSet = set()
    for i in range(len(annotations)):
        if annotations[i]=="Positive":
            positiveSet.add(text[i])
        elif annotations[i]=="Negative":
            negativeSet.add(text[i])
    return positiveSet,negativeSet

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
#For WordCloud
def wordcounter(text):
    return(len(text))

def word_frequent(text):
    fd_content = FreqDist(text)
    return fd_content

def save_wordcloud(text,fileName):
    wc = WordCloud(width=1000, height=600, background_color="white", random_state=0)
    plt.imshow(wc.generate_from_frequencies(text))
    plt.axis("off")
    plt.savefig("sentimentAnalysis/static/img/"+fileName+".png", format = "png")

#For Surface Metric
def top_freqeunt(list):
    fd_content = FreqDist(list)
    return fd_content.most_common(5)

def sentenceLevel(text):
    result = []
    for tweet in text:
        result.append(nltk.sent_tokenize(tweet))
    return result

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

def stanfordNLPSentimentFunction(sentences):
    nlp = StanfordCoreNLP('http://localhost:9000')
    result=[]
    count_pol = []
    for i in range(0,len(sentences)):
        positive = 0
        negative = 0
        neutral = 0
        cnt=[0,0,0,0,0]
        res = nlp.annotate(sentences[i],properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 50000,
                   })
        for s in res["sentences"]:
            cnt[int(s["sentimentValue"])]+=1
        index=cnt.index(max(cnt))
        if index == (1 or 2):
          negative += 1
          count_pol.append([positive, neutral, negative])
        elif index == (3 or 4):
          positive += 1
          count_pol.append([positive, neutral, negative])
        else:
            neutral += 1
            count_pol.append([positive, neutral, negative])
        result.append(getPolarity(index))
    return result, count_pol

def stanfordNLPSentimentFunction_sentence(sentences):
    polarities = []
    count_polarity = []
    count = len(sentences)
    for i in range(count):
        polarity, count_pol = stanfordNLPSentimentFunction(sentences[i])
        polarities.append(polarity)
        count_polarity.append(count_pol)
    return polarities, count_polarity

#switch use dictionary
def getPolarity(x):
    return {
        0: "Negative",
        1: "Negative",
        2: "Neutral",
        3: "Positive",
        4: "Positive",
    }[x]

def textblobSentimentFunction(sentences):
    scores = []
    polarities = []
    count_pol = []
    for sentence in sentences:
        positive = 0
        negative = 0
        neutral = 0
        testimonial = TextBlob(sentence)
        score = testimonial.sentiment.polarity
        if score >= 0.05:
            polarities.append("Positive")
            positive += 1
            count_pol.append([positive, neutral, negative])
        elif score <0.05 and score >-0.05:
            polarities.append("Neutral")
            neutral += 1
            count_pol.append([positive, neutral, negative])
        elif score<=-0.05:
            polarities.append("Negative")
            negative += 1
            count_pol.append([positive, neutral, negative])
        scores.append(round(score,2))
    return scores, polarities, count_pol

def textblobSentimentFunction_sentence(sentences):
    scores=[]
    polarities = []
    count_polarity = []
    count = len(sentences)
    for i in range(count):
        score, polarity, count_pol = textblobSentimentFunction(sentences[i])
        scores.append(score)
        polarities.append(polarity)
        count_polarity.append(count_pol)
    return scores, polarities, count_polarity

#For sentiwordnet function
def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def sentiWordnetSentimentFunction(text):
    scores = []
    polarities = []
    tokens_count = 0
    count_pol=[]

    for sentence in text:
        sentiment = 0.0
        raw_sentences = sent_tokenize(sentence)
        positive = 0
        negative = 0
        neutral = 0
        #품사 태그
        for raw_sentence in raw_sentences:
            tagged_sentence = pos_tag(word_tokenize(raw_sentence))

            for word, tag in tagged_sentence:
                wn_tag = penn_to_wn(tag)
                if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB):
                    continue

                lemma = WordNetLemmatizer().lemmatize(word, pos=wn_tag)
                if not lemma:
                    continue

                synsets = wn.synsets(lemma, pos=wn_tag)
                if not synsets:
                    continue

                # Take the first sense, the most common
                synset = synsets[0]
                swn_synset = swn.senti_synset(synset.name())

                sentiment += swn_synset.pos_score() - swn_synset.neg_score()
                tokens_count += 1

        scores.append(round(sentiment,2))
        if sentiment > 0:
            polarities.append("Positive")
            positive += 1
            count_pol.append([positive, neutral, negative])
        elif sentiment < 0:
            polarities.append("Negative")
            negative += 1
            count_pol.append([positive, neutral, negative])
        else:
            polarities.append("Neutral")
            neutral += 1
            count_pol.append([positive, neutral, negative])
    return scores, polarities, count_pol

def sentiWordnetSentimentFunction_sentence(sentences):
    scores=[]
    polarities = []
    count_polarity = []
    count = len(sentences)
    for i in range(count):
        score, polarity, count_pol = sentiWordnetSentimentFunction(sentences[i])
        scores.append(score)
        polarities.append(polarity)
        count_polarity.append(count_pol)
    return scores, polarities, count_polarity

def confusionMatrix (annotation_result, tool_result):
 return confusion_matrix(annotation_result, tool_result, labels=["Positive", "Negative"])

def precise(annotation_result, tool_result):
    return precision_score(annotation_result, tool_result, average='macro')

def recall(annotation_result, tool_result):
    return recall_score(annotation_result, tool_result, average='macro')

def F1Score(precision, recall):
    F1Score = 2*precision*recall/(precision+recall)
    return F1Score

def F1ScoreSorted(vaderF1,textBlobF1,sentiF1):
    result=[("Vader F1 Score",vaderF1),("TextBlob",textBlobF1),("SentiWordNet Dictionary F1 Score",sentiF1)]
    result.sort(key=lambda element : element[1],reverse=True)
    return result

def average(score):
    result = []
    count = len(score)
    for i in range(count):
        sum = 0
        for j in score[i]:
            sum += j
        result.append(round(sum/count,2))
    return result

def majority(polarities):
    result = []
    count = len(polarities)
    for i in range(count):
        pos = 0
        neg = 0
        for polarity in polarities[i]:
            if polarity == 'Positive':
                pos += 1
            elif polarity == 'Negative':
                neg += 1
            else:
                continue
        if pos == neg:
            result.append("Neutral")
        elif pos > neg:
            result.append("Positive")
        else:
            result.append("Negative")
    return result

def sum_for_kappa_sentence(a,b,c,d):
  count_pol = []
  for i in range(len(a)):
    count_polarity=[]
    for j in range(len(a[i])):
      sum_pos = a[i][j][0]+b[i][j][0]+c[i][j][0]+d[i][j][0]
      sum_neu = a[i][j][1]+b[i][j][1]+c[i][j][1]+d[i][j][1]
      sum_neg = a[i][j][2]+b[i][j][2]+c[i][j][2]+d[i][j][2]
      count_polarity.append([sum_pos, sum_neu, sum_neg])
    count_pol.append(count_polarity)
  return count_pol

def sum_for_kappa_tweet(a,b,c,d):
    result = []
    for i in range(len(a)):
      count_pol = []
      sum_pos = a[i][0]+b[i][0]+c[i][0]+d[i][0]
      sum_neu = a[i][1]+b[i][1]+c[i][1]+d[i][1]
      sum_neg = a[i][2]+b[i][2]+c[i][2]+d[i][2]
      count_pol.append([sum_pos, sum_neu, sum_neg])
      result.append(count_pol)
    return result

def fleiss_kappa(matrixes):
    result = []
    for matrix in matrixes:
        M = np.array(matrix)
        N, k = M.shape  # N is # of items, k is # of categories
        n_annotators = float(np.sum(M[0]))
        p = np.sum(M, axis=0) / (N * n_annotators)
        P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
        Pbar = np.sum(P) / N
        PbarE = np.sum(p * p)
        kappa = ((Pbar - PbarE) / (1 - PbarE))
        result.append(round(kappa,2))
    return result
