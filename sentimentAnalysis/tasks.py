from __future__ import absolute_import, unicode_literals
from celery import shared_task
from .models import Request,tweet,requestResult,sentenceResult,tasklist
import random,os,re,time
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from textblob import TextBlob

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

from pycorenlp import StanfordCoreNLP
from nltk.corpus import stopwords

from nltk import FreqDist
from wordcloud import WordCloud
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score
import numpy as np
import json
from os import path

import smtplib #module for SMTP
from email.mime.text import MIMEText #settings for messages title and content

import yaml
stream = open("sentimentAnalysis/mailKey.yml", 'r')
key = yaml.safe_load(stream)

import pandas as pd

@shared_task
def run(name,email):
    try:
        print("Process ID : " + str(os.getpid()))
        Request.objects.filter(request_name = name, request_owner = email).update(request_status = "pending",request_pid = os.getpid())
        request = Request.objects.filter(request_name = name, request_owner = email).first()
        request_id = request.key
        filePath = request.file_path
        file = open(filePath, "r", encoding='utf-8',errors="ignore")
        text=file.readlines()
        ids = []
        content = []
        hashtag = []
        annotation = []

        pattern = '#([0-9a-zA-Z]*)'
        hashtag_word = re.compile(pattern)

        for line in text:
            sentence = re.split(r'\t+', line)
            text = ""
            ids.append(sentence[0])
            content.append(sentence[1])
            annotation.append(sentence[2].strip('\n'))

            for tag in hashtag_word.findall(line):
                hashtag.append(tag)

        for i in range(0,len(ids)):
            tweet(key=None,requestName=name,userEmail = email,tweet_id=ids[i],tweet_content=content[i],tweet_annotation=annotation[i],vaderScores=0.0,vaderPolarity='vpolarity',textblobScores=.0,textblobPolarity='tpolarity',sentiWordNetScores=0.0,
            sentiWordNetPolarity='spolarity',stanfordNLPPolarity='nlppolarity',kappa=0.0).save()

        content_sentence = sentenceLevel(content) #by sentence

        for i in range(0,len(ids)):
            for j in range(0,len(content_sentence[i])):
                sentenceResult(key=None,requestName=name,userEmail = email,tweet_id=ids[i],sentenceID=j,sentence_content=content_sentence[i][j],vaderScores=0.0,vaderPolarity='vpolarity',textblobScores=.0,textblobPolarity='tpolarity',sentiWordNetScores=0.0,
                sentiWordNetPolarity='spolarity',stanfordNLPPolarity='nlppolarity').save()

        requestResult(key=None,requestName=name, userEmail = email,vaderConfusionMatrix="confusion", vaderPrecise=150.0, vaderRecall=150.0, vaderF1Score=150.0,textblobConfusionMatrix="confusion", textblobPrecise=150.0, textblobRecall=150.0, textblobF1Score=150.0,sentiWordNetConfusionMatrix="confusion", sentiWordNetPrecise=150.0, sentiWordNetRecall=150.0, sentiWordNetF1Score=150.0,stanfordNLPConfusionMatrix="confusion",
        topFrequentWords ="topFrequentWords",wordCounter=0,wordCloudFileName="wordCloudFileName",hashtagFrequent="hashtagFrequent",positiveTopFrequentHashtag="positiveTopFrequentHashtag",negativeTopFrequentHashtag="negativeTopFrequentHashtag",positiveTopFrequentWords="positiveTopFrequentWords",positiveWordcounter=0,positiveWordCloudFilename="positiveWordCloudFilename",negativeTopFrequentWords="negativeTopFrequentWords",negativeWordcounter=0,
        frequentwordFilename = "frequentwordFilename",   frequentHashtagFilename = "frequentHashtagFilename", positive_frequentwordFilename= "positive_frequentwordFilename", positive_frequentHashtagFilename = "positive_frequentHashtagFilename", negative_frequentwordFilename = "negative_frequentwordFilename", negative_frequentHashtagFilename = "negative_frequentHashtagFilename",
        negativeWordCloudFilename="negativeWordCloudFilename",sortedF1ScoreList="sortedF1ScoreList",vaderCountpol="",textblobCountpol="",sentiWordNetCountpol ="", stanfordNLPCountpol="",tweetIDs='',wordGraphFilename="").save()

        vaderAnalysis.delay(name,email, ids, content,annotation,content_sentence,request_id)
        textblobAnalysis.delay(name,email, ids, content,annotation,content_sentence,request_id)
        sentiWordNetAnalysis.delay(name,email, ids, content,annotation,content_sentence,request_id)
        stanfordNLPAnalysis.delay(name,email, ids, content,annotation,content_sentence,request_id)

        cleansingText = cleansing(content)
        word_frequents = word_frequent(cleansingText)
        topFrequentWords=top_freqeunt(cleansingText)
        wordcounters = wordcounter(cleansingText)
        filenames = str(name)+ "_"+str(email) + "_WordCloud"
        wordGraphFileNames = str(name)+ "_"+str(email) + "_WordGraph"
        frequntwordfilename = str(name)+ "_"+str(email) + "_FrequentWord"
        frequenthashtagfilename = str(name)+ "_"+str(email) + "_FrequentHashtag"
        save_wordcloud(word_frequents,filenames)
        hashtag_frequent = top_freqeunt(hashtag)
        frequent_graph(topFrequentWords, frequntwordfilename)
        frequent_hashtag_graph(hashtag_frequent, frequenthashtagfilename)

        positiveSet,negativeSet=separatePN(annotation,content)
        positiveList = list(positiveSet)
        negativeList = list(negativeSet)
        positiveHashtag= extractHashtag(positiveList)
        negativeHashtag= extractHashtag(negativeList)
        positiveCleansingText = cleansing(positiveList)
        positiveWordCounter = wordcounter(positiveCleansingText)
        positiveWord_frequent = word_frequent(positiveCleansingText)
        save_wordcloud(positiveWord_frequent,filenames+"_Positive")
        negativeCleansingText = cleansing(negativeList)
        negativeWordCounter = wordcounter(negativeCleansingText)
        negativeWord_frequent = word_frequent(negativeCleansingText)
        save_wordcloud(negativeWord_frequent,filenames+"_Negative")
        word_graph(wordcounters, positiveWordCounter, negativeWordCounter, wordGraphFileNames)
        positivefrequentHashtag = word_frequent(positiveHashtag)
        negativefrequentHashtag = word_frequent(negativeHashtag)
        frequent_graph(positiveWord_frequent,frequntwordfilename+"_Positive")
        frequent_hashtag_graph(positivefrequentHashtag,frequntwordfilename+"_Positive")
        frequent_graph(negativeWord_frequent,frequntwordfilename+"_Negative")
        frequent_hashtag_graph(negativefrequentHashtag,frequntwordfilename+"_Negative")

        req = requestResult.objects.filter(requestName=name, userEmail = email).update(topFrequentWords = topFrequentWords,wordCounter =wordcounters, wordCloudFileName = "img/"+ filenames +".png", hashtagFrequent = hashtag_frequent, positiveTopFrequentHashtag=top_freqeunt(positiveHashtag),negativeTopFrequentHashtag=top_freqeunt(negativeHashtag),positiveTopFrequentWords=top_freqeunt(positiveCleansingText),positiveWordcounter = positiveWordCounter, wordGraphFilename ="img/"+ wordGraphFileNames+".png", positiveWordCloudFilename = "img/"+filenames+"_Positive.png", negativeTopFrequentWords=top_freqeunt(negativeCleansingText),negativeWordcounter = negativeWordCounter,negativeWordCloudFilename = "img/"+filenames+"_Negative.png",
        frequentwordFilename = "img/"+ frequntwordfilename +".png", frequentHashtagFilename= "img/"+ frequenthashtagfilename +".png", positive_frequentwordFilename= "img/"+ frequntwordfilename +"_Positive.png", positive_frequentHashtagFilename= "img/"+ frequenthashtagfilename +"_Positive.png", negative_frequentwordFilename= "img/"+ frequntwordfilename +"_Negative.png", negative_frequentHashtagFilename= "img/"+ frequenthashtagfilename +"_Negative.png" , tweetIDs = str(ids))
        print("SUCCESS : ",str(os.getpid()))
        if os.path.exists(filePath):
            os.remove(filePath)
    except Exception as exception:
        print(exception)
        request.request_status="failure"
        request.save()


@shared_task
def vaderAnalysis(requestName,email,tweet_id, tweet_content, tweet_annotation,content_sentence,request_id):
        try:
            print("Vader Process ID : " + str(os.getpid()))
            tasklist.objects.filter(request_key = request_id, toolName ="vader").update(toolStatus = "pending", tool_pid = os.getpid())

            analyzer = SentimentIntensityAnalyzer()
            polarities = []
            vaderScore = []
            vaderCount = []
            for i in range(0,len(tweet_id)):
                positive = 0
                negative = 0
                neutral = 0
                vs = analyzer.polarity_scores(tweet_content[i])
                vaderScore.append(round(vs['compound'],2))
                if vs['compound'] >= 0.05:
                    polarities.append("Positive")
                    positive += 1
                    vaderCount.append([positive, neutral, negative])
                elif vs['compound'] <0.05 and vs['compound'] >-0.05:
                    polarities.append("Neutral")
                    neutral += 1
                    vaderCount.append([positive, neutral, negative])
                elif vs['compound']<=-0.05:
                    polarities.append("Negative")
                    negative += 1
                    vaderCount.append([positive, neutral, negative])

            vaderScores_sentence, vaderPolarity_sentence, vaderCountpol_sentence = vaderSentimentFucntion_sentence(content_sentence)

            vaderAverage = average(vaderScores_sentence)
            vaderMajority = majority(vaderPolarity_sentence)

            for i in range(0,len(tweet_id)):
                for j in range(0,len(content_sentence[i])):
                    sentenceResult.objects.filter(requestName=requestName,userEmail=email,tweet_id=tweet_id[i],sentenceID=j).update(vaderScores = vaderScores_sentence[i][j], vaderPolarity = vaderPolarity_sentence[i][j])

            for i in range(0,len(tweet_id)):
                tweet.objects.filter(requestName=requestName,userEmail=email,tweet_id=tweet_id[i]).update(vaderScores = vaderScore[i], vaderPolarity = polarities[i],vaderAverage=vaderAverage[i],vaderMajority = vaderMajority[i])

            precise = round(precision_score(tweet_annotation, polarities, average='macro'),2)
            recall = round(recall_score(tweet_annotation, polarities, average='macro'),2)

            requestResult.objects.filter(requestName=requestName,userEmail = email).update(vaderConfusionMatrix = str(confusion_matrix(tweet_annotation,polarities,labels=["Positive", "Negative"])),vaderPrecise = precise,vaderRecall = recall, vaderF1Score = round(2*precise*recall/(precise+recall),2),vaderCountpol = str(vaderCount),vaderCountpol_sentence = str(vaderCountpol_sentence))

            tasklist.objects.filter(request_key = request_id, toolName ="vader").update(toolStatus = "success")

            vader = tasklist.objects.filter(request_key = request_id, toolName ="vader").first()
            text = tasklist.objects.filter(request_key = request_id, toolName ="textblob").first()
            senti = tasklist.objects.filter(request_key = request_id, toolName ="sentiWordNet").first()
            stan = tasklist.objects.filter(request_key = request_id, toolName ="stanfordNLP").first()

        except NameError as exception:
            tasklist.objects.filter(request_key = request_id, toolName ="vader").update(toolStatus = "failure")
            print("Vader import Error")

@shared_task
def textblobAnalysis(requestName,email,tweet_id, tweet_content, tweet_annotation,content_sentence,request_id):
    try:
        print("TextBlob Process ID : " + str(os.getpid()))
        tasklist.objects.filter(request_key = request_id, toolName ="textblob").update(toolStatus = "pending", tool_pid = os.getpid())

        scores = []
        polarities = []
        count_pol = []
        for i in range(0,len(tweet_id)):
            positive = 0
            negative = 0
            neutral = 0
            testimonial = TextBlob(tweet_content[i])
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

        textblobScores_sentence, textblobPolarity_sentence, textblobCountpol_sentence = textblobSentimentFunction_sentence(content_sentence)
        textblobAverage = average(textblobScores_sentence)
        textblobMajority = majority(textblobPolarity_sentence)

        for i in range(0,len(tweet_id)):
            for j in range(0,len(content_sentence[i])):
                sentenceResult.objects.filter(requestName=requestName,userEmail=email, tweet_id=tweet_id[i],sentenceID=j).update(textblobScores = textblobScores_sentence[i][j], textblobPolarity = textblobPolarity_sentence[i][j])

        for i in range(0,len(tweet_id)):
            tweet.objects.filter(requestName=requestName,userEmail=email,tweet_id=tweet_id[i]).update(textblobScores = scores[i], textblobPolarity = polarities[i],textblobAverage=textblobAverage[i],textblobMajority = textblobMajority[i])

        result = requestResult.objects.get(requestName=requestName)
        precise = round(precision_score(tweet_annotation, polarities, average='macro'),2)
        recall = round(recall_score(tweet_annotation, polarities, average='macro'),2)

        requestResult.objects.filter(requestName=requestName,userEmail = email).update(textblobConfusionMatrix = str(confusion_matrix(tweet_annotation,polarities,labels=["Positive", "Negative"])),textblobPrecise = precise,textblobRecall = recall, textblobF1Score = round(2*precise*recall/(precise+recall),2),textblobCountpol = str(count_pol),textblobCountpol_sentence = str(textblobCountpol_sentence))
        tasklist.objects.filter(request_key = request_id, toolName ="textblob").update(toolStatus = "success")

        vader = tasklist.objects.filter(request_key = request_id, toolName ="vader").first()
        text = tasklist.objects.filter(request_key = request_id, toolName ="textblob").first()
        senti = tasklist.objects.filter(request_key = request_id, toolName ="sentiWordNet").first()
        stan = tasklist.objects.filter(request_key = request_id, toolName ="stanfordNLP").first()

    except NameError as exception:
        tasklist.objects.filter(request_key = request_id, toolName ="textblob").update(toolStatus = "failure")
        print("Vader import Error")

@shared_task
def sentiWordNetAnalysis(requestName,email,tweet_id, tweet_content, tweet_annotation,content_sentence,request_id):
    try:
        print("SentiWordNet Process ID : " + str(os.getpid()))
        tasklist.objects.filter(request_key = request_id, toolName ="sentiWordNet").update(toolStatus = "pending", tool_pid = os.getpid())

        scores = []
        polarities = []
        tokens_count = 0
        count_pol=[]

        for i in range(0,len(tweet_id)):
            sentiment = 0.0
            raw_sentences = sent_tokenize(tweet_content[i])
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

        sentiWordnetScore_sentence, sentiWordnetPolarity_sentence, sentiWordnetCountpol_sentence = sentiWordnetSentimentFunction_sentence(content_sentence)
        sentiWordnetAverage = average(sentiWordnetScore_sentence)
        sentiWordnetMajority = majority(sentiWordnetPolarity_sentence)

        for i in range(0,len(tweet_id)):
            for j in range(0,len(content_sentence[i])):
                sentenceResult.objects.filter(requestName=requestName,userEmail=email,tweet_id=tweet_id[i],sentenceID=j).update(sentiWordNetScores = sentiWordnetScore_sentence[i][j], sentiWordNetPolarity = sentiWordnetPolarity_sentence[i][j])

        for i in range(0,len(tweet_id)):
            tweet.objects.filter(requestName=requestName,userEmail=email,tweet_id=tweet_id[i]).update(sentiWordNetScores = scores[i], sentiWordNetPolarity = polarities[i],sentiWordnetAverage=sentiWordnetAverage[i],sentiWordnetMajority = sentiWordnetMajority[i])

        result = requestResult.objects.get(requestName=requestName)
        precise = round(precision_score(tweet_annotation, polarities, average='macro'),2)
        recall = round(recall_score(tweet_annotation, polarities, average='macro'),2)

        requestResult.objects.filter(requestName=requestName,userEmail = email).update(sentiWordNetConfusionMatrix = str(confusion_matrix(tweet_annotation,polarities,labels=["Positive", "Negative"])),sentiWordNetPrecise = precise,sentiWordNetRecall = recall, sentiWordNetF1Score = round(2*precise*recall/(precise+recall),2),sentiWordNetCountpol = str(count_pol),sentiWordnetCountpol_sentence = str(sentiWordnetCountpol_sentence))
        tasklist.objects.filter(request_key = request_id, toolName ="sentiWordNet").update(toolStatus = "success")

        vader = tasklist.objects.filter(request_key = request_id, toolName ="vader").first()
        text = tasklist.objects.filter(request_key = request_id, toolName ="textblob").first()
        senti = tasklist.objects.filter(request_key = request_id, toolName ="sentiWordNet").first()
        stan = tasklist.objects.filter(request_key = request_id, toolName ="stanfordNLP").first()

    except NameError as exception:
        tasklist.objects.filter(userEmail = email,requestName=requestName, toolName ="sentiWordNet").update(toolStatus = "failure")
        print("SentiWordNet import Error")

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

@shared_task
def stanfordNLPAnalysis(requestName,email,tweet_id, tweet_content, tweet_annotation,content_sentence,request_id):
    try:
        print("StanfordNLP Process ID : " + str(os.getpid()))
        tasklist.objects.filter(request_key = request_id, toolName ="stanfordNLP").update(toolStatus = "pending", tool_pid = os.getpid())

        nlp = StanfordCoreNLP('http://localhost:9000')
        result=[]
        count_pol = []
        for i in range(0,len(tweet_id)):
            positive = 0
            negative = 0
            neutral = 0
            cnt=[0,0,0,0,0]
            res = nlp.annotate(tweet_content[i],properties={
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

        stanfordNLPPolarity_sentence, stanfordNLPCountpol_sentence = stanfordNLPSentimentFunction_sentence(content_sentence)
        stanfordNLPMajority = majority(stanfordNLPPolarity_sentence)

        for i in range(0,len(tweet_id)):
            for j in range(0,len(content_sentence[i])):
                sentenceResult.objects.filter(requestName=requestName,userEmail=email,tweet_id=tweet_id[i],sentenceID=j).update(stanfordNLPPolarity = stanfordNLPPolarity_sentence[i][j])

        for i in range(0,len(tweet_id)):
            tweet.objects.filter(requestName=requestName,userEmail=email,tweet_id=tweet_id[i]).update(stanfordNLPPolarity = result[i],stanfordNLPMajority = stanfordNLPMajority[i])

        requestRes = requestResult.objects.get(requestName=requestName)

        requestResult.objects.filter(requestName=requestName,userEmail = email).update(stanfordNLPConfusionMatrix = str(confusion_matrix(tweet_annotation,result,labels=["Positive", "Negative"])),stanfordNLPCountpol =str(count_pol),stanfordNLPCountpol_sentence = str(stanfordNLPCountpol_sentence))
        tasklist.objects.filter(request_key = request_id, toolName ="stanfordNLP").update(toolStatus = "success")

        vader = tasklist.objects.filter(request_key = request_id, toolName ="vader").first()
        text = tasklist.objects.filter(request_key = request_id, toolName ="textblob").first()
        senti = tasklist.objects.filter(request_key = request_id, toolName ="sentiWordNet").first()
        stan = tasklist.objects.filter(request_key = request_id, toolName ="stanfordNLP").first()

    except NameError as exception:
        tasklist.objects.filter(request_key = request_id, toolName ="stanfordNLP").update(toolStatus = "failure")
        print("Stanford NLP import Error")
    except Exception as exception:
        print(exception)
        tasklist.objects.filter(request_key = request_id, toolName ="stanfordNLP").update(toolStatus = "failure")
        print("Check Stanford NLP Server")

@shared_task
def totalAnalysis(requestName, email,tweet_id):
    result = requestResult.objects.filter(requestName=requestName,userEmail = email).first()
    requestResult.objects.filter(requestName=requestName,userEmail = email).update(sortedF1ScoreList = str(F1ScoreSorted(result.vaderF1Score,result.textblobF1Score,result.sentiWordNetF1Score)))

    sumPolarity_sentence = sum_for_kappa_sentence(result.vaderCountpol_sentence, result.textblobCountpol_sentence, result.sentiWordnetCountpol_sentence, result.stanfordNLPCountpol_sentence)
    sumPolarity_tweet = sum_for_kappa_tweet(result.vaderCountpol, result.textblobCountpol, result.sentiWordNetCountpol, result.stanfordNLPCountpol)
    KappaScore_sentence = fleiss_kappa(sumPolarity_sentence)
    kappas=fleiss_kappa(sumPolarity_tweet)
    for i in range(0,len(tweet_id)):
        tweet.objects.filter(requestName=requestName,userEmail = email,tweet_id=tweet_id[i]).update(kappa = kappas[i],sentenceKappa = KappaScore_sentence[i])

    Request.objects.filter(request_name = requestName, request_owner = email).update(request_status = "success",request_completed_time = time.strftime(r"%Y-%m-%d %H:%M:%S", time.localtime()))

    sendMail(email)

def sendMail(email):
    print("Mail Sending Code")
    s = smtplib.SMTP('smtp.gmail.com',587)
    s.starttls() #setting about tls mode, tls : transport layer security

    s.login('uci.salt.sender@gmail.com',key['mailKey'])

    msg = MIMEText('This is the content of Sentiment Analysis Mail reporting.')
    msg['Subject'] = 'Sentiment Analysis Mail Reporting test'

    s.sendmail("uci.salt.sender@gmail.com",email,msg.as_string())
    s.quit()

#switch use dictionary
def getPolarity(x):
    return {
        0: "Negative",
        1: "Negative",
        2: "Neutral",
        3: "Positive",
        4: "Positive",
    }[x]

def wordcounter(text):
    return(len(text))

def word_frequent(text):
    fd_content = FreqDist(text)
    return fd_content

def save_wordcloud(text,fileName):
    wc = WordCloud(width=1000, height=600, background_color="white", random_state=0)
    wc.generate_from_frequencies(text)
    wc.to_file(path.join("sentimentAnalysis/static/img/", fileName+".png"))

#For Surface Metric
def top_freqeunt(list):
    fd_content = FreqDist(list)
    return fd_content.most_common(5)

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

def F1ScoreSorted(vaderF1,textBlobF1,sentiF1):
    result=[("Vader F1 Score",vaderF1),("TextBlob",textBlobF1),("SentiWordNet Dictionary F1 Score",sentiF1)]
    result.sort(key=lambda element : element[1],reverse=True)
    return result

def sum_for_kappa_sentence(a,b,c,d):
    a = json.loads(a)
    b = json.loads(b)
    c = json.loads(c)
    d = json.loads(d)
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
    a = json.loads(a)
    b = json.loads(b)
    c = json.loads(c)
    d = json.loads(d)
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
    try:
        for matrix in matrixes:
            M = np.array(matrix)
            N, k = M.shape  # N is # of items, k is # of categories
            n_annotators = float(np.sum(M[0]))
            p = np.sum(M, axis=0) / (N * n_annotators)
            P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
            Pbar = np.sum(P) / N
            PbarE = np.sum(p * p)
            kappa = (Pbar - PbarE) / (1 - PbarE)
            if Pbar == 1.0:
                result.append(1.0)
            else:
                 result.append(round(kappa,2))
    except Exception as exception:
        print(exception)
        print("this is KAPPA")
    return result

def average(score):
    result = []
    for i in score:
        avg = sum(i, 0.0)/len(i)
        result.append(round(avg,2))
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

def sentenceLevel(text):
    result = []
    for tweet in text:
        result.append(nltk.sent_tokenize(tweet))
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

def word_graph(wordconter, positiveWordcounter, negativeWordcounter,wordGraphFileName):
    matplotlib.use("Agg")
    colors = ['green', 'orange', 'red']
    pos_per = round((positiveWordcounter/wordconter)*100)
    neg_per = round((negativeWordcounter/wordconter)*100)
    neu_per = round(100-pos_per-neg_per)
    a = np.array([[pos_per, neu_per, neg_per]])
    df = pd.DataFrame(a, columns= ['positive', 'neutral', 'negative'])
    ax = df.plot.barh(color = colors, stacked=True, figsize = (9, 1.5), edgecolor = "none")
    fig1 = plt.gcf()
    for p in ax.patches:
        left, bottom, width, height = p.get_bbox().bounds
        ax.annotate((width), xy=(left+width/2, bottom+height/2), ha='center', va='center', fontsize = 18)
    plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1), edgecolor = "none")
    plt.sca(ax)
    plt.box(False)
    plt.axis('off')
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
    plt.draw()
    fig1.savefig("sentimentAnalysis/static/img/"+wordGraphFileName+'.png')
    plt.close(fig1)

#frequnt_graph
def frequent_graph(list, frequentWordFilename):
    matplotlib.use("Agg")
    keyword = []
    value = []
    for i in list:
        keyword.append(i[0])
        value.append(i[1])
    fig= plt.gcf()
    plt.barh(keyword, value)
    plt.draw()
    fig.savefig("sentimentAnalysis/static/img/"+frequentWordFilename+'.png')
    plt.close(fig)


def frequent_hashtag_graph(list, frequentHashtahFilename):
    matplotlib.use("Agg")
    keyword = []
    value = []
    for i in list:
        keyword.append(i[0])
        value.append(i[1])
    fig= plt.gcf()
    plt.barh(keyword, value)
    plt.draw()
    fig.savefig("sentimentAnalysis/static/img/"+frequentHashtahFilename+'.png')
    plt.close(fig)
