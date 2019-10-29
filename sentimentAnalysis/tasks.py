from __future__ import absolute_import, unicode_literals
from celery import shared_task
from .models import Requestlist,tweet,requestResult
import random,os,re,time
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from textblob import TextBlob

from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

from pycorenlp import StanfordCoreNLP


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
    sentiWordNetAnalysis.delay(id, ids, content,annotation)
    stanfordNLPAnalysis.delay(id, ids, content,annotation)

    request.save()
    print("SUCCESS : ",str(os.getpid()))

@shared_task
def vaderAnalysis(request_id,tweet_id, tweet_content, tweet_annotation):
    print("Vader Process ID : " + str(os.getpid()))
    request = Requestlist.objects.get(request_id=request_id)
    request.vader_status = "pending"
    request.vader_pid = os.getpid()
    request.save()

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

    request = Requestlist.objects.get(request_id=request_id)
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
    request.save()

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

    request = Requestlist.objects.get(request_id=request_id)
    request.textblob_status = "success"
    request.save()

    #Checker 비동기적으로 짜면 수정할 코드
    if request.vader_status == "success" and request.textblob_status == "success" and request.sentiWordNet_status =="success" and request.stanfordNLP_status == "success":
        #mail보내기 코드
        request.request_status = "success"
        request.request_completed_time = time.strftime(r"%Y-%m-%d %H:%M:%S", time.localtime())
        request.save()

@shared_task
def sentiWordNetAnalysis(request_id,tweet_id, tweet_content, tweet_annotation):
    print("SentiWordNet Process ID : " + str(os.getpid()))
    request = Requestlist.objects.get(request_id=request_id)
    request.sentiWordNet_status = "pending"
    request.sentiWordNet_pid = os.getpid()
    request.save()

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
    for i in range(0,len(tweet_id)):
        tweet.objects.filter(request_id=request_id,tweet_id=tweet_id[i]).update(sentiWordNetScores = scores[i], sentiWordNetPolarity = polarities[i], sentiWordNetCountpol = count_pol[i])

    result = requestResult.objects.get(request_id=request_id)
    result.sentiWordNetConfusionMatrix = str(confusion_matrix(tweet_annotation,polarities,labels=["Positive", "Negative"]))
    precise = round(precision_score(tweet_annotation, polarities, average='macro'),2)
    result.sentiWordNetPrecise = precise
    recall = round(recall_score(tweet_annotation, polarities, average='macro'),2)
    result.sentiWordNetRecall = recall
    result.sentiWordNetF1Score = round(2*precise*recall/(precise+recall),2)
    result.save()

    request = Requestlist.objects.get(request_id=request_id)
    request.sentiWordNet_status = "success"
    request.save()

    #Checker 비동기적으로 짜면 수정할 코드
    if request.vader_status == "success" and request.textblob_status == "success" and request.sentiWordNet_status =="success" and request.stanfordNLP_status == "success":
        #mail보내기 코드
        request.request_status = "success"
        request.request_completed_time = time.strftime(r"%Y-%m-%d %H:%M:%S", time.localtime())
        request.save()

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
def stanfordNLPAnalysis(request_id,tweet_id, tweet_content, tweet_annotation):
    print("StanfordNLP Process ID : " + str(os.getpid()))
    request = Requestlist.objects.get(request_id=request_id)
    request.stanfordNLP_status = "pending"
    request.stanfordNLP_pid = os.getpid()
    request.save()

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

    for i in range(0,len(tweet_id)):
        tweet.objects.filter(request_id=request_id,tweet_id=tweet_id[i]).update(stanfordNLPPolarity = result[i], stanfordNLPCountpol = count_pol[i])

    requestRes = requestResult.objects.get(request_id=request_id)
    #print("이게 뭔데? : ", str(confusion_matrix(tweet_annotation,result,labels=["Positive", "Negative"])))
    requestRes.stanfordNLPConfusionMatrix = str(confusion_matrix(tweet_annotation,result,labels=["Positive", "Negative"]))
    requestRes.save()

    request = Requestlist.objects.get(request_id=request_id)
    request.stanfordNLP_status = "success"
    request.save()

    #Checker 비동기적으로 짜면 수정할 코드
    if request.vader_status == "success" and request.textblob_status == "success" and request.sentiWordNet_status =="success" and request.stanfordNLP_status == "success":
        #mail보내기 코드
        request.request_status = "success"
        request.request_completed_time = time.strftime(r"%Y-%m-%d %H:%M:%S", time.localtime())
        request.save()

#switch use dictionary
def getPolarity(x):
    return {
        0: "Negative",
        1: "Negative",
        2: "Neutral",
        3: "Positive",
        4: "Positive",
    }[x]
