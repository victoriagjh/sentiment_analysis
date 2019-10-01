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

from .models import Requestlist, RequestlistManager

def main(request):
    if request.method == 'POST':
        if 'user_id' in request.POST:
            Requestlist(user_id = request.POST.get('user_id', ''), project_number = request.POST.get('project_number',0), project_status = "processing").save()
            lists = Requestlist.objects.all()
            context = {'lists':lists}
        
        return render(request, "expert_page.html", context)
        
    return render(request, "main_page.html")# context)


            
