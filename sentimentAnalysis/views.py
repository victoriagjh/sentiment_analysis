from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

from django.shortcuts import render
from django.http import HttpResponseRedirect
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

# Create your views here.
def sentimentAnalysis(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            tools=[]
            if type(request.POST.get('vader')) !=type(None):
                tools.append("Vader Sentiment")
            if type(request.POST.get('textblob')) != type(None):
                tools.append("TextBlob")
            if tools:
                #preprocess the file
                form.save()
                form.name = request.FILES['file'].name
                text = handle_uploaded_file(request.FILES['file'])

                ids,contents,annotations = preprocessFile(text)
                form.ids=ids
                form.annotations=convertSentimentResult("userFile",annotations)
                form.text = text #id, text, annotation type(list)
                form.content = contents #text type(list)
                form.tool = tools

                #all content type(str)
                contentString=""
                for i in contents:
                    contentString+=i

                #surface metrics
                cleansingText = cleansing(contentString)
                form.word_frequent = word_frequent(cleansingText)
                form.topFrequentWords=top_freqeunt(cleansingText)
                form.wordcounter = wordcounter(cleansingText)
                save_wordcloud(form.word_frequent)

                #sentimentAnalysis
                for i in tools:
                    if i == "Vader Sentiment":
                        form.vaderScores=vaderSentimentFucntion(form.content)
                        form.vaderPolarity=convertSentimentResult("Vader",form.vaderScores)
                        form.vaderCategory=compareFileWithVader(form.annotations,form.vaderPolarity)
                    elif i == "TextBlob":
                        form.textblobScores=textblobSentimentFunction(form.content)
                        form.textblobPolarity=convertSentimentResult("TextBlob",form.textblobScores)
                        form.textblobCategory=compareFileWithVader(form.annotations,form.textblobPolarity)
                context = {
                    'form':form,
                    }
                return render(request, "result_page.html",context)
            if not tools:
                messages.warning(request, 'You should check the tool at least 1!', extra_tags='alert')
    else:
        form = UploadFileForm()
    context = {
        'form':form,
    }
    return render(request, 'main_page.html', context)


def result_page(request):
    if request.method == 'POST':
        if 'spacy_button' in request.POST:
            context = {
            'spacy':"spacy",
            }
            return render(request, 'last.html', context)
    return render(request,'main_page.html',context)

def handle_uploaded_file(f):
    fileName="text/"+f.name
    file = open(fileName, "r")
    text=file.readlines()
    textList=[]
    for i in text:
        i=i.replace('\n','\t')
        i=i.replace('   ','\t')
        i=i.split('\t')
        for j in i:
            if j!='':
                textList.append(j)
    file.close()
    return textList

def preprocessFile(text):
    content=[]
    id=[]
    annotation=[]
    s=0
    for i in text:
        if s%3==1 and s!=1:
            content.append(i)
        elif s%3==0 and s!=0:
            id.append(i)
        elif s%3==2 and s!=2:
            annotation.append(i)
        s+=1
    return id,content,annotation

def cleansing(text):
    lower_content = (text.lower())

    shortword = re.compile(r'\W*\b\w{1,2}\b')
    shortword_content = shortword.sub('', lower_content)
    text = re.sub('[-=.#/?:$}!,]', '', shortword_content)

    stop_words = set(stopwords.words('english'))
    content_tokens = word_tokenize(text)
    result = []

    for w in content_tokens:
        if w not in stop_words:
            result.append(w)
    return result

def wordcounter(text):
    return(len(text))

def word_frequent(text):
    fd_content = FreqDist(text)
    return fd_content

def top_freqeunt(text):
    fd_content = FreqDist(text)
    return fd_content.most_common(5)

def save_wordcloud(text):
    wc = WordCloud(width=1000, height=600, background_color="white", random_state=0)
    plt.imshow(wc.generate_from_frequencies(text))
    plt.axis("off")
    plt.savefig("sentimentAnalysis/static/img/test.png", format = "png")

def vaderSentimentFucntion(sentences):
    analyzer = SentimentIntensityAnalyzer()
    result = []
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        result.append(vs['compound']) #only Compound value
    return result

#convert the result sentiment analysis for compare each of things
def convertSentimentResult(toolName,sentimentResult):
    converted=[]
    if toolName=="Vader" or toolName=="TextBlob":
        for i in sentimentResult:
            if i >= 0.05:
                converted.append("positive")
            elif i<0.05 and i>-0.05:
                converted.append("neutral")
            elif i<=-0.05:
                converted.append("negative")
        return converted
    elif toolName=="userFile":
        for i in sentimentResult:
            if i == "4":
                converted.append("positive")
            elif i == "2":
                converted.append("neutral")
            elif i == "0":
                converted.append("negative")
        return converted
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
    result = []
    for sentence in sentences:
        testimonial = TextBlob(sentence)
        result.append(testimonial.sentiment.polarity)
    return result
