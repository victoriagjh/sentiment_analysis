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

from .models import Request, RequestlistManager,tasklist
import time
from .tasks import run
import random
import yaml
import pyrebase

session ={}
stream = open("sentimentAnalysis/config.yml", 'r')
data_loaded = yaml.safe_load(stream)

#for firebase
config =  data_loaded

#Initialize Firebase
firebase = pyrebase.initialize_app(config)
auther = firebase.auth()
database = firebase.database()

def main(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid() and 'request_id' in request.POST:
            form.save() #Save the Input File
            filePath="text/"+request.FILES['file'].name
            Request(key=None, request_name = request.POST.get('request_id', ''), request_owner = request.POST.get('request_owner',0), request_status = "unassigned", request_pid = 0,
            request_issued_time = time.strftime(r"%Y-%m-%d %H:%M:%S", time.localtime()),request_completed_time = time.strftime(r"%Y-%m-%d %H:%M:%S", time.localtime()), file_path = filePath).save()
            unassignedRequest = None
            unassignedRequest = Request.objects.filter(request_name = request.POST.get('request_id', ''), request_owner = request.POST.get('request_owner', '')).first()

            tasklist(key=None, request_key = unassignedRequest.key, toolName = "vader",toolStatus="unassigned", tool_pid = 0).save()
            tasklist(key=None, request_key = unassignedRequest.key, toolName = "textblob",toolStatus="unassigned", tool_pid = 0).save()
            tasklist(key=None, request_key = unassignedRequest.key, toolName = "sentiWordNet",toolStatus="unassigned", tool_pid = 0).save()
            tasklist(key=None, request_key = unassignedRequest.key, toolName = "stanfordNLP",toolStatus="unassigned", tool_pid = 0).save()
            if unassignedRequest != None :
                try:
                    run.apply_async(kwargs={'name': request.POST.get('request_id', ''), 'email': request.POST.get('request_owner', '')},time_limit=60*30, soft_time_limit=60*30)
                except SoftTimeLimitExceeded:
                    print("SoftTimeLimitExceeded : ", SoftTimeLimitExceeded)
                    clean_up_in_a_hurry()
                except TimeoutError as err:
                    print("Timeout Error : ", err)
                #run.delay(unassignedRequest.request_id)
            lists = Request.objects.all()
            context = {'lists':lists}
            return render(request, "expert_page.html", context)
    return render(request, "main_page.html") # context)
#이지 코드 넣은 부분
def google_login(request):
    return render(request, "google.html")

def afterlogin(request):
    return  render(request, "main_page.html")

#로그인 페이지 들어가기
def signIn(request):
    return render(request, "loginpage.html")

#로그인 진행
def postsign(request):
    email = request.POST.get('email')
    password = request.POST.get('password')
    try:
        user = auther.sign_in_with_email_and_password(email, password)
        print('user: ', user)
    except Exception as exception:
        print("ERROR : ", exception)
        messages = "invalid credentials"
        return render(request, "loginpage.html", {"message":messages})
    #print(user['idToken'])
    session_id=user['idToken']
    request.session['uid'] = str(session_id)

    return render(request, "main_page.html", {"e":email})

def logout_view(request):
    auth.logout(request)
    return render(request, "main_page.html")

#회원가입 창
def signUp(request):
    return render(request, 'signUp.html')

#회원가입진행
def postsignup(request):
    name = request.POST.get('name')
    email = request.POST.get('email')
    password = request.POST.get('password')
    try:
        user = auther.create_user_with_email_and_password(email, password)
        print("user:" , user)
    except Exception as exception:
        print("ERROR : ", exception)
        messages = "unable to create account try again"
        return render(request, "signUp.html", {"message":messages})

    uid = user['localId']
    data = {"name":name, "status":"l"}
    database.child("users").child(uid).child("details").set(data)

    return render(request, "loginpage.html")
