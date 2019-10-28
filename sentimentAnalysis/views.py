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
import time
from .tasks import run
import random

def main(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid() and 'request_id' in request.POST:
            form.save() #Save the Input File
            filePath="text/"+request.FILES['file'].name
            Requestlist(request_id = request.POST.get('request_id', ''), request_owner = request.POST.get('request_owner',0), request_status = "unassigned", request_pid = 0,
            vader_status="unassigned", vader_pid = 0, textblob_status = "unassigned",textblob_pid = 0, stanfordNLP_status= "unassigned",stanfordNLP_pid = 0, sentiWordNet_status="unassigned", sentiWordNet_pid = 0,
            request_issued_time = time.strftime(r"%Y-%m-%d %H:%M:%S", time.localtime()),request_completed_time = time.strftime(r"%Y-%m-%d %H:%M:%S", time.localtime()), file_path = filePath).save()
            unassignedRequest = None
            unassignedRequest = Requestlist.objects.get(request_id = request.POST.get('request_id', ''))
            if unassignedRequest != None :
                try:
                    run.apply_async(kwargs={'id': unassignedRequest.request_id},time_limit=60*30, soft_time_limit=60*30)
                except SoftTimeLimitExceeded:
                    print("SoftTimeLimitExceeded : ", SoftTimeLimitExceeded)
                    clean_up_in_a_hurry()
                except TimeoutError as err:
                    print("Timeout Error : ", err)
                #run.delay(unassignedRequest.request_id)
            lists = Requestlist.objects.all()
            context = {'lists':lists}
            return render(request, "expert_page.html", context)
    return render(request, "main_page.html") # context)
#이지 코드 넣은 부분
def afterlogin(request):
    return  render(request, "main_page.html")

def signIn(request):
    return render(request, "loginpage.html")

def postsign(request):
    email = request.POST.get('email')
    password = request.POST.get('password')
    try:
        user = auther.sign_in_with_email_and_password(email, password)
    except:
        messages = "invalid credentials"
        return render(request, "signIn.html", {"message":messages})
    print(user['idToken'])
    session_id=user['idToken']
    request.session['uid'] = str(session_id)
    return render(request, "main_page.html")

def logout_view(request):
    logout(request)
    return render(request, "main_page.html")

def logout(request):
    auth.logout(request)
    return render(request, 'loginpage.html')

def signUp(request):
    return render(request, 'signUp.html')

def postsignup(request):
    name = request.POST.get('name')
    email = request.POST.get('email')
    password = request.POST.get('password')
    try:
        user = auther.create_user_with_email_and_password(email, password)
    except:
        messages = "unable to create account try again"
        return render(request, "signUp.html", {"message":messages})

    uid = user['localId']
    data = {"name":name, "status":"l"}
    database.child("users").child(uid).child("details").set(data)
    return render(request, "signIn.html")
