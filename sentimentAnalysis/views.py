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

from .models import Request,tasklist,tweet,requestResult,sentenceResult
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
import pyrebase
from django.contrib import auth

firebase = pyrebase.initialize_app(config)
auther = firebase.auth()
database = firebase.database()

def main(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid() and 'request_id' in request.POST:
            overlapRequest = Request.objects.filter(request_name = request.POST.get('request_id', ''), request_owner = request.POST.get('request_owner', '')).first()
            if overlapRequest != None:
                messages.info(request, 'Your Request Name is overlapped. Please Try Again.')
                return HttpResponseRedirect('/')
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
            return render(request, "request_submitted.html", context)
                #try login
        if(request.POST.get('email') != None and request.POST.get('password') != None):
            email = request.POST.get('email')
            password = request.POST.get('password')
            try:
                user = auther.sign_in_with_email_and_password(email, password)
                print('user: ', user)  #for checking
                return render(request, "main_page.html", {"e": email})
            except Exception as exception:
                print("ERROR : ", exception)
                messages = "invalided account"
                return render(request, "loginpage.html", {"message":messages})
    return render(request, "main_page.html") # context)

def signIn(request):
    return render(request, "loginpage.html")

def logout_view(request):
    response = render(request, "main_page.html")
    response.delete_cookie('token_info')
    auth.logout(request)
    return render(request, "main_page.html")

def signUp(request):
    if(request.POST.get('email') != None):
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        try:
            user = auther.create_user_with_email_and_password(email, password)
            print('user: ', user)

        except Exception as exception:
            print("ERROR : ", exception)
            messages = "unable to create account try again"
            return render(request, "signUp.html", {"message":messages})

        uid = user['localId']
        data = {"name":name, "status":"l"}
        database.child("users").child(uid).child("details").set(data)

        return render(request, "loginpage.html")
    return render(request, 'signUp.html')

def history(request):
    #Look the cookie and use the JWT, decode it
    # request.POST.get('request_owner', '')
    '''
    Request.objects.all().delete()
    tasklist.objects.all().delete()
    tweet.objects.all().delete()
    requestResult.objects.all().delete()
    sentenceResult.objects.all().delete()
    '''
    requestList = Request.objects.filter(request_owner = 'akrso06197@naver.com').order_by('request_issued_time')
    form = {'requestList' : requestList}
    print(requestList)
    return render(request,"history.html", form)

def requestDetail(request,request_owner,request_name):
    performance = requestResult.objects.filter(userEmail = request_owner, requestName = request_name).first()
    print(performance)
    tweets = tweet.objects.filter(userEmail = request_owner, requestName = request_name).order_by('kappa')
    sentences = sentenceResult.objects.filter(userEmail = request_owner, requestName = request_name)
    context = {'performance':performance, 'tweets': tweets, 'sentences': sentences}
    return render(request, "requestDetail.html", context)

def requestExplorer(request,request_owner,request_name):
    requests = requestResult.objects.filter(userEmail = request_owner, requestName = request_name).first()
    tweets = tweet.objects.filter(userEmail = request_owner, requestName = request_name).order_by('kappa')
    sentences = sentenceResult.objects.filter(userEmail = request_owner, requestName = request_name)
    context = {'requests': requests,'tweets': tweets, 'sentences': sentences}
    return render(request, "explorer_page.html", context)
