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

# Create your views here.
def sentimentAnalysis(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            tools=[]
            if type(request.POST.get('nltk')) !=type(None):
                tools.append(request.POST.get('nltk'))
            if type(request.POST.get('spacy')) != type(None):
                tools.append(request.POST.get('spacy'))
            if type(request.POST.get('scikit')) != type(None):
                tools.append(request.POST.get('scikit'))
            if type(request.POST.get('r')) != type(None):
                tools.append(request.POST.get('r'))
            if type(request.POST.get('bagOfWords')) != type(None):
                tools.append(request.POST.get('bagOfWords'))
            if tools:
                form.save()
                form.name = request.FILES['file'].name
                text = handle_uploaded_file(request.FILES['file'])
                content = getContent(text)
                form.text = text
                form.content = content #list형태 
                form.tool = tools

                contents=""
                for i in content:
                    contents+=i

                cleansingText = cleansing(contents)
                form.word_frequent = word_frequent(cleansingText)
                form.topFrequentWords=top_freqeunt(cleansingText)
                form.wordcounter = wordcounter(cleansingText)
                save_wordcloud(form.word_frequent)

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

def getContent(text):
    content=[]
    s=0
    for i in text:
        if s%3==1 and s!=1:
            content.append(i)
        s+=1
    return content

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