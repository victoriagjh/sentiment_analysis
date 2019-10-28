from __future__ import absolute_import, unicode_literals
from celery import shared_task
from .models import Requestlist,tweet,requestResult
import random,os,re,time
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from textblob import TextBlob


@shared_task
def run(id):
    print("Process ID : " + str(os.getpid()))
    request = Requestlist.objects.get(request_id=id)
    request.request_status = "pending"
    request.request_pid = os.getpid()
    request.save()
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
    #check

    for i in range(0,len(ids)):
        tweet(key=None,request_id=id,tweet_id=ids[i],tweet_content=content[i],tweet_annotation=annotation[i],vaderScores=0.0,vaderPolarity='vpolarity',vaderCountpol="",textblobScores=.0,textblobPolarity='tpolarity',textblobCountpol="",sentiWordNetScores=0.0,
        sentiWordNetPolarity='spolarity',sentiWordNetCountpol="",stanfordNLPPolarity='nlppolarity',stanfordNLPCountpol="").save()

    requestResult(request_id=id, vaderConfusionMatrix="confusion", vaderPrecise=150.0, vaderRecall=150.0, vaderF1Score=150.0,textblobConfusionMatrix="confusion", textblobPrecise=150.0, textblobRecall=150.0, textblobF1Score=150.0,sentiWordNetConfusionMatrix="confusion", sentiWordNetPrecise=150.0, sentiWordNetRecall=150.0, sentiWordNetF1Score=150.0,stanfordNLPConfusionMatrix="confusion").save()
    vaderAnalysis.delay(id, ids, content,annotation)
    textblobAnalysis.delay(id, ids, content,annotation)


    request.save()
    print("SUCCESS : ",str(os.getpid()))

@shared_task
def vaderAnalysis(request_id,tweet_id, tweet_content, tweet_annotation):
    print("Vader Process ID : " + str(os.getpid()))
    request = Requestlist.objects.get(request_id=request_id)
    request.vader_status = "pending"
    request.vader_pid = os.getpid()

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
            vaderCount.append(str([positive, neutral, negative]))
        elif vs['compound'] <0.05 and vs['compound'] >-0.05:
            polarities.append("Neutral")
            neutral += 1
            vaderCount.append(str([positive, neutral, negative]))
        elif vs['compound']<=-0.05:
            polarities.append("Negative")
            negative += 1
            vaderCount.append(str([positive, neutral, negative]))

    for i in range(0,len(tweet_id)):
        tweet.objects.filter(request_id=request_id,tweet_id=tweet_id[i]).update(vaderScores = vaderScore[i], vaderPolarity = polarities[i], vaderCountpol = vaderCount[i])

    result = requestResult.objects.get(request_id=request_id)
    result.vaderConfusionMatrix = str(confusion_matrix(tweet_annotation,polarities,labels=["Positive", "Negative"]))
    precise = round(precision_score(tweet_annotation, polarities, average='macro'),2)
    result.vaderPrecise = precise
    recall = round(recall_score(tweet_annotation, polarities, average='macro'),2)
    result.vaderRecall = recall
    result.vaderF1Score = round(2*precise*recall/(precise+recall),2)
    result.save()

    request.vader_status = "success"
    request.save()

    #Checker 비동기적으로 짜면 수정할 코드
    if request.vader_status == "success" and request.textblob_status == "success" and request.sentiWordNet_status =="success" and request.stanfordNLP_status == "success":
        #mail보내기 코드
        request.request_status = "success"
        request.request_completed_time = time.strftime(r"%Y-%m-%d %H:%M:%S", time.localtime())
        request.save()

@shared_task
def textblobAnalysis(request_id,tweet_id, tweet_content, tweet_annotation):
    print("TextBlob Process ID : " + str(os.getpid()))
    request = Requestlist.objects.get(request_id=request_id)
    request.textblob_status = "pending"
    request.textblob_pid = os.getpid()

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

    for i in range(0,len(tweet_id)):
        tweet.objects.filter(request_id=request_id,tweet_id=tweet_id[i]).update(textblobScores = scores[i], textblobPolarity = polarities[i], textblobCountpol = count_pol[i])

    result = requestResult.objects.get(request_id=request_id)
    result.textblobConfusionMatrix = str(confusion_matrix(tweet_annotation,polarities,labels=["Positive", "Negative"]))
    precise = round(precision_score(tweet_annotation, polarities, average='macro'),2)
    result.textblobPrecise = precise
    recall = round(recall_score(tweet_annotation, polarities, average='macro'),2)
    result.textblobRecall = recall
    result.textblobF1Score = round(2*precise*recall/(precise+recall),2)
    result.save()

    request.textblob_status = "success"
    request.save()
