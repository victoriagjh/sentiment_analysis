from django.db import models
from django.core.validators import FileExtensionValidator
from datetime import datetime
from django.contrib.auth.base_user import AbstractBaseUser, BaseUserManager

# Create your models here.
class UploadFileModel(models.Model):
    file = models.FileField(validators=[FileExtensionValidator(allowed_extensions=['txt'])])

class RequestlistManager(models.Manager):
    def create_request(self, request_id, request_owner, request_status,request_pid,vader_status,vader_pid,textblob_status,textblob_pid,stanfordNLP_status,stanfordNLP_pid,sentiWordNet_status,sentiWordNet_pid,request_issued_time,request_completed_time,file_path):
        request= self.create(request_id = user_number, request_owner = request_owner, request_status = request_status, request_pid = request_pid,vader_status=vader_status, vader_pid = vader_pid, textblob_status = textblob_status,textblob_pid = textblob_pid, stanfordNLP_status= stanfordNLP_status,stanfordNLP_pid = stanfordNLP_pid,sentiWordNet_status=sentiWordNet_status,sentiWordNet_pid = sentiWordNet_pid,request_issued_time = request_issued_time,request_completed_time = request_completed_time, file_path = file_path)
        return request

class Requestlist(models.Model):
    request_id = models.TextField(default = "something", primary_key=True)
    request_owner = models.TextField(default = "something")
    request_status = models.TextField(default = "unassigned")
    request_pid = models.IntegerField(default=0)
    vader_status = models.TextField(default = "unassigned")
    vader_pid = models.IntegerField(default=0)
    textblob_status = models.TextField(default = "unassigned")
    textblob_pid = models.IntegerField(default=0)
    stanfordNLP_status = models.TextField(default = "unassigned")
    stanfordNLP_pid =  models.IntegerField(default=0)
    sentiWordNet_status = models.TextField(default = "unassigned")
    sentiWordNet_pid = models.IntegerField(default=0)
    request_issued_time = models.DateTimeField(default=datetime.now, blank=True)
    request_completed_time = models.DateTimeField(default=datetime.now, blank=True)
    file_path = models.TextField(default = "/")
    objects = RequestlistManager()

class tweetResultManager(models.Manager):
    def create_request(self, key, request_id, tweet_id, tweet_content, tweet_annotation, vaderScores,vaderPolarity,textblobScores,textblobPolarity, sentiWordNetScores,sentiWordNetPolarity,stanfordNLPPolarity,kappa,vaderAverage,vaderMajority,textblobAverage,textblobMajority,sentiWordnetAverage, sentiWordnetMajority, stanfordNLPMajority,sentenceKappa):
        request = self.create(key=key, request_id = request_id, tweet_id=tweet_id, tweet_content=tweet_content, tweet_annotation=tweet_annotation, vaderScores=vaderScores,vaderPolarity=vaderPolarity, textblobScores=textblobScores,textblobPolarity=textblobPolarity, sentiWordNetScores=sentiWordNetScores,sentiWordNetPolarity=sentiWordNetPolarity,
        stanfordNLPPolarity = stanfordNLPPolarity,kappa=kappa, vaderAverage=vaderAverage,vaderMajority=vaderMajority,textblobAverage=textblobAverage,textblobMajority=textblobMajority,sentiWordnetAverage=sentiWordnetAverage, sentiWordnetMajority=sentiWordnetMajority, stanfordNLPMajority=stanfordNLPMajority,sentenceKappa=sentenceKappa)
        return request

class tweet(models.Model):
    key = models.AutoField(primary_key=True,auto_created=True)
    request_id = models.TextField(default = "something")
    tweet_id = models.TextField(default = "tweet_id")
    tweet_content = models.TextField(default = "tweet_content")
    tweet_annotation = models.TextField(default = "tweet_annotation")
    vaderScores = models.FloatField(default=150.0)
    vaderPolarity = models.CharField(max_length=200,default='SOME STRING')

    textblobScores = models.FloatField(default=150.0)
    textblobPolarity = models.CharField(max_length=200,default='SOME STRING')

    sentiWordNetScores = models.FloatField(default=150.0)
    sentiWordNetPolarity = models.CharField(max_length=200,default='SOME STRING')

    stanfordNLPPolarity = models.CharField(max_length=200,default='SOME STRING')

    kappa = models.FloatField(default=150.0)

    vaderAverage = models.FloatField(default=150.0)
    vaderMajority = models.TextField(default = "something")
    textblobAverage = models.FloatField(default=150.0)
    textblobMajority = models.TextField(default = "something")
    sentiWordnetAverage = models.FloatField(default=150.0)
    sentiWordnetMajority = models.TextField(default = "something")
    stanfordNLPMajority = models.TextField(default = "something")

    sentenceKappa = models.FloatField(default=150.0)

    objects = tweetResultManager()

class requestResultManager(models.Manager):
    def create_request(self, request_id, vaderConfusionMatrix,vaderPrecise,vaderRecall,vaderF1Score, textblobConfusionMatrix,textblobPrecise,textblobRecall,textblobF1Score, sentiWordNetConfusionMatrix,sentiWordNetPrecise,sentiWordNetRecall,sentiWordNetF1Score, stanfordNLPConfusionMatrix,
    topFrequentWords,wordCounter,wordCloudFileName,hashtagFrequent,positiveTopFrequentHashtag,negativeTopFrequentHashtag,positiveTopFrequentWords,positiveWordcounter,positiveWordCloudFilename,negativeTopFrequentWords,negativeWordcounter,negativeWordCloudFilename,sortedF1ScoreList,vaderCountpol,textblobCountpol,sentiWordNetCountpol,stanfordNLPCountpol,vaderCountpol_sentence ,textblobCountpol_sentence ,sentiWordNetCountpol_sentence ,stanfordNLPCountpol_sentence ):
        request = self.create(request_id = request_id, vaderConfusionMatrix=vaderConfusionMatrix,vaderPrecise=vaderPrecise,vaderRecall=vaderRecall,vaderF1Score=vaderF1Score, textblobConfusionMatrix=textblobConfusionMatrix,textblobPrecise=textblobPrecise,textblobRecall=textblobRecall,textblobF1Score=textblobF1Score, sentiWordNetConfusionMatrix=sentiWordNetConfusionMatrix,sentiWordNetPrecise=sentiWordNetPrecise,
        sentiWordNetRecall=sentiWordNetRecall,sentiWordNetF1Score=sentiWordNetF1Score, stanfordNLPConfusionMatrix=stanfordNLPConfusionMatrix,
        topFrequentWords =topFrequentWords,wordCounter=wordCounter,wordCloudFileName=wordCloudFileName,hashtagFrequent=hashtagFrequent,positiveTopFrequentHashtag=positiveTopFrequentHashtag,negativeTopFrequentHashtag=negativeTopFrequentHashtag,positiveTopFrequentWords=positiveTopFrequentWords,positiveWordcounter=positiveWordcounter,positiveWordCloudFilename=positiveWordCloudFilename,negativeTopFrequentWords=negativeTopFrequentWords,negativeWordcounter=negativeWordcounter,
        negativeWordCloudFilename=negativeWordCloudFilename,sortedF1ScoreList=sortedF1ScoreList, vaderCountpol=vaderCountpol,textblobCountpol=textblobCountpol,sentiWordNetCountpol=sentiWordNetCountpol,stanfordNLPCountpol=stanfordNLPCountpol,vaderCountpol_sentence=vaderCountpol_sentence ,textblobCountpol_sentence=textblobCountpol_sentence ,sentiWordNetCountpol_sentence=sentiWordNetCountpol_sentence ,stanfordNLPCountpol_sentence =stanfordNLPCountpol_sentence)
        return request

class requestResult(models.Model): #Request당 나오는 결과물
    request_id = models.TextField(default = "something")

    vaderConfusionMatrix = models.CharField(max_length=200,default='SOME STRING')
    vaderPrecise = models.FloatField(default=150.0)
    vaderRecall = models.FloatField(default=150.0)
    vaderF1Score = models.FloatField(default=150.0)

    textblobConfusionMatrix = models.CharField(max_length=200,default='SOME STRING')
    textblobPrecise = models.FloatField(default=150.0)
    textblobRecall = models.FloatField(default=150.0)
    textblobF1Score = models.FloatField(default=150.0)

    sentiWordNetConfusionMatrix = models.CharField(max_length=200,default='SOME STRING')
    sentiWordNetPrecise = models.FloatField(default=150.0)
    sentiWordNetRecall = models.FloatField(default=150.0)
    sentiWordNetF1Score = models.FloatField(default=150.0)

    stanfordNLPConfusionMatrix = models.CharField(max_length=200,default='SOME STRING')

    topFrequentWords = models.CharField(max_length=200,default='SOME STRING')
    wordCounter = models.IntegerField(default=0)
    wordCloudFileName = models.CharField(max_length=200,default='SOME STRING')

    hashtagFrequent = models.CharField(max_length=200,default='SOME STRING')
    positiveTopFrequentHashtag = models.CharField(max_length=200,default='SOME STRING')
    negativeTopFrequentHashtag = models.CharField(max_length=200,default='SOME STRING')

    positiveTopFrequentWords = models.CharField(max_length=200,default='SOME STRING')
    positiveWordcounter = models.IntegerField(default=0)
    positiveWordCloudFilename = models.CharField(max_length=200,default='SOME STRING')

    negativeTopFrequentWords = models.CharField(max_length=200,default='SOME STRING')
    negativeWordcounter = models.IntegerField(default=0)
    negativeWordCloudFilename = models.CharField(max_length=200,default='SOME STRING')

    sortedF1ScoreList = models.CharField(max_length=200,default='SOME STRING')

    vaderCountpol = models.CharField(max_length=10000000,default='SOME STRING')
    textblobCountpol = models.CharField(max_length=10000000,default='SOME STRING')
    sentiWordNetCountpol = models.CharField(max_length=10000000,default='SOME STRING')
    stanfordNLPCountpol = models.CharField(max_length=10000000,default='SOME STRING')

    vaderCountpol_sentence = models.CharField(max_length=10000000,default='SOME STRING')
    textblobCountpol_sentence = models.CharField(max_length=10000000,default='SOME STRING')
    sentiWordnetCountpol_sentence = models.CharField(max_length=10000000,default='SOME STRING')
    stanfordNLPCountpol_sentence = models.CharField(max_length=10000000,default='SOME STRING')

    objects = requestResultManager()

class sentenceResultManager(models.Manager):
    def create_request(self, key, request_id, tweet_id, sentenceID,tweet_content, tweet_annotation, vaderScores,vaderPolarity,textblobScores,textblobPolarity, sentiWordNetScores,sentiWordNetPolarity,stanfordNLPPolarity):
        request = self.create(key=key, request_id = request_id, tweet_id=tweet_id, sentenceID = sentenceID,tweet_content=tweet_content, tweet_annotation=tweet_annotation, vaderScores=vaderScores,vaderPolarity=vaderPolarity, textblobScores=textblobScores,textblobPolarity=textblobPolarity, sentiWordNetScores=sentiWordNetScores,sentiWordNetPolarity=sentiWordNetPolarity,
        stanfordNLPPolarity = stanfordNLPPolarity)
        return request

class sentenceResult(models.Model):
    key = models.AutoField(primary_key=True,auto_created=True)
    request_id = models.TextField(default = "something")
    tweet_id = models.TextField(default = "tweet_id")
    sentenceID =  models.IntegerField(default=0)

    vaderScores = models.FloatField(default=150.0)
    vaderPolarity = models.CharField(max_length=200,default='SOME STRING')
    textblobScores = models.FloatField(default=150.0)
    textblobPolarity = models.CharField(max_length=200,default='SOME STRING')
    sentiWordNetScores = models.FloatField(default=150.0)
    sentiWordNetPolarity = models.CharField(max_length=200,default='SOME STRING')
    stanfordNLPPolarity = models.CharField(max_length=200,default='SOME STRING')

    objects = sentenceResultManager()

class TokenInfoManager(models.Manager):
    def create_token_info(self, uid, email, token, state):
        request = self.create(uid=uid, email=email, token=token, state=state)
        return request

class TokenInfo(models.Model):
    uid = models.TextField(default = "something")
    email = models.TextField(default = "something")
    token = models.TextField(default = "something")
    state = models.TextField(default="1")