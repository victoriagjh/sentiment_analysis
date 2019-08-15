from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

import os
from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponseRedirect,HttpResponse,Http404
from .forms import UploadFileForm
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
                form.content = contents #text type(list)
                form.content_sentence = sentenceLevel(form.content)
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
                        form.vaderScores, form.vaderPolarity =vaderSentimentFucntion(form.content)
                        form.vaderScores_sentence, form.vaderPolarity_sentence = vaderSentimentFucntion_sentence(form.content_sentence)
                        form.vaderAverage = average(form.vaderScores_sentence)
                        form.vaderMajority = majority(form.vaderPolarity_sentence)
                        #form.vaderCategory=compareFileWithVader(form.annotations,form.vaderPolarity)
                        form.vaderConfusionMatrix = confusionMatrix(form.annotations, form.vaderPolarity)
                        form.vaderPrecise = precise(form.annotations, form.vaderPolarity)
                        form.vaderRecall = recall(form.annotations, form.vaderPolarity)
                        form.vaderF1Score = F1Score(form.vaderPrecise,form.vaderRecall)
                    elif i == "TextBlob":
                        form.textblobScores, form.textblobPolarity =textblobSentimentFunction(form.content)
                        form.textblobScores_sentence, form.textblobPolarity_sentence = textblobSentimentFunction_sentence(form.content_sentence)
                        form.textblobAverage = average(form.textblobScores_sentence)
                        form.textblobMajority = majority(form.textblobPolarity_sentence)
                        #form.textblobCategory=compareFileWithVader(form.annotations,form.textblobPolarity)
                        form.textblobConfusionMatrix = confusionMatrix(form.annotations, form.textblobPolarity)
                        form.textblobPrecise = precise(form.annotations, form.textblobPolarity)
                        form.textblobRecall = recall(form.annotations, form.textblobPolarity)
                        form.textblobF1Score = F1Score(form.textblobPrecise,form.textblobRecall)
                    elif i == "sentiWordnet":
                        form.sentiWordnetScore, form.sentiWordnetPolarity = sentiWordnetSentimentFunction(form.content)
                        form.sentiWordnetScore_sentence, form.sentiWordnetPolarity_sentence = sentiWordnetSentimentFunction_sentence(form.content_sentence)
                        form.sentiWordnetAverage = average(form.sentiWordnetScore_sentence)
                        form.sentiWordnetMajority = majority(form.sentiWordnetPolarity_sentence)
                        form.sentiWordnetConfusionMatrix = confusionMatrix(form.annotations, form.sentiWordnetPolarity)
                        form.sentiWordnetPrecise = precise(form.annotations, form.sentiWordnetPolarity)
                        form.sentiWordnetRecall = recall(form.annotations, form.sentiWordnetPolarity)
                        form.sentiWordnetF1Score = F1Score(form.textblobPrecise,form.sentiWordnetRecall)
                    elif i == "Stanford NLP":
                        form.stanfordNLPPolarity = stanfordNLPSentimentFunction(form.content)
                        form.stanfordNLPPolarity_sentence = stanfordNLPSentimentFunction_sentence(form.content_sentence)
                        form.stanfordNLPMajority = majority(form.stanfordNLPPolarity_sentence)
                        form.stanfordNLPConfusionMatrix = confusionMatrix(form.annotations, form.stanfordNLPPolarity)

                positiveSet,negativeSet=separatePN(form.annotations,form.content)
                positiveList = list(positiveSet)
                negativeList = list(negativeSet)
                positiveHashtag= extractHashtag(positiveList)
                negativeHashtag= extractHashtag(negativeList)
                form.positiveTopFrequentHashtag=top_freqeunt(positiveHashtag)
                form.negativeTopFrequentHashtag=top_freqeunt(negativeHashtag)

                #sorted Sentiment Result F1 Score
                form.sortedF1ScoreLists = F1ScoreSorted(form.vaderF1Score,form.textblobF1Score,form.sentiWordnetF1Score)

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
        if 'metric' in request.POST:
            return render(request,'expert_metrics.html',session)
        if 'download' in request.POST:
            makeFile("expert")
            path_to_file = os.path.realpath("result.txt")
            file = open(path_to_file, 'r')
            rfile = File(file)
            response = HttpResponse(rfile, content_type='application/txt')
            response['Content-Disposition'] = 'attachment; filename=' + "result.txt"
            return response
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

def convertToString(list):
    contentString=""
    for i in list:
        contentString+=i
        contentString+=" "
    return contentString

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

def wordcounter(text):
    return(len(text))

def word_frequent(text):
    fd_content = FreqDist(text)
    return fd_content

def top_freqeunt(list):
    fd_content = FreqDist(list)
    return fd_content.most_common(5)

def save_wordcloud(text,fileName):
    wc = WordCloud(width=1000, height=600, background_color="white", random_state=0)
    plt.imshow(wc.generate_from_frequencies(text))
    plt.axis("off")
    plt.savefig("sentimentAnalysis/static/img/"+fileName+".png", format = "png")

def sentenceLevel(text):
    result = []
    for tweet in text:
        result.append(nltk.sent_tokenize(tweet))
    return result

def vaderSentimentFucntion(sentences):
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    polarities = []
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        scores.append(vs['compound']) #only Compound value
        if vs['compound'] >= 0.05:
            polarities.append("Positive")
        elif vs['compound'] <0.05 and vs['compound'] >-0.05:
            polarities.append("Neutral")
        elif vs['compound']<=-0.05:
            polarities.append("Negative")
    return scores, polarities


def vaderSentimentFucntion_sentence(sentences):
    scores = []
    polarities = []
    count = len(sentences)
    for i in range(count):
        score, polarity = vaderSentimentFucntion(sentences[i])
        scores.append(score)
        polarities.append(polarity)
    return scores, polarities

#switch use dictionary
def getPolarity(x):
    return {
        0: "Negative",
        1: "Negative",
        2: "Neutral",
        3: "Positive",
        4: "Positive",
    }[x]

def stanfordNLPSentimentFunction(sentences):
    nlp = StanfordCoreNLP('http://localhost:9000')
    result=[]
    for i in range(0,len(sentences)):
        cnt=[0,0,0,0,0]
        res = nlp.annotate(sentences[i],properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 50000,
                   })
        for s in res["sentences"]:
            cnt[int(s["sentimentValue"])]+=1
        index=cnt.index(max(cnt))
        result.append(getPolarity(index))
    return result

def stanfordNLPSentimentFunction_sentence(sentences):
    polarities = []
    count = len(sentences)
    for i in range(count):
        polarity = stanfordNLPSentimentFunction(sentences[i])
        polarities.append(polarity)
    return polarities


#convert the result sentiment analysis for compare each of things
def convertSentimentResult(toolName,sentimentResult):
    converted=[]
    if toolName=="Vader" or toolName=="TextBlob":
        for i in sentimentResult:
            if i >= 0.05:
                converted.append("Positive")
            elif i<0.05 and i>-0.05:
                converted.append("Neutral")
            elif i<=-0.05:
                converted.append("Negative")
        return converted
    elif toolName=="userFile":
        for i in sentimentResult:
            if i == "4":
                converted.append("Positive")
            elif i == "2":
                converted.append("Neutral")
            elif i == "0":
                converted.append("Negative")
        return converted
    elif toolName == "sentiWordnet":
        for i in sentimentResult:
            if i > 0:
                converted.append("Positive")
            elif i < 0:
                converted.append("Negative")
            else:
                converted.append("Neutral")
    return converted

#compare the polarity of two lists
def compareFileWithVader(annotations, toolResults):
    category=[]
    for i in range(0,len(annotations)):
        if annotations[i]==toolResults[i]:
            category.append(True)
        else:
            category.append(False)
    return category

def textblobSentimentFunction(sentences):
    scores = []
    polarities = []
    for sentence in sentences:
        testimonial = TextBlob(sentence)
        score = testimonial.sentiment.polarity
        if score >= 0.05:
            polarities.append("Positive")
        elif score <0.05 and score >-0.05:
            polarities.append("Neutral")
        elif score<=-0.05:
            polarities.append("Negative")
        scores.append(score)
    return scores, polarities

def textblobSentimentFunction_sentence(sentences):
    scores=[]
    polarities = []
    count = len(sentences)
    for i in range(count):
        score, polarity = textblobSentimentFunction(sentences[i])
        scores.append(score)
        polarities.append(polarity)
    return scores, polarities

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

    for sentence in text:
        sentiment = 0.0
        raw_sentences = sent_tokenize(sentence)
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

        scores.append(sentiment)
        if sentiment > 0:
            polarities.append("Positive")
        elif sentiment < 0:
            polarities.append("Negative")
        else:
            polarities.append("Neutral")
    return scores, polarities

def sentiWordnetSentimentFunction_sentence(sentences):
    scores=[]
    polarities = []
    count = len(sentences)
    for i in range(count):
        score, polarity = sentiWordnetSentimentFunction(sentences[i])
        scores.append(score)
        polarities.append(polarity)
    return scores, polarities

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
        result.append(sum/count)
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

def kappaScore(annotation_result, tool_result):
    return cohen_kappa_score(annotation_result, tool_result)
