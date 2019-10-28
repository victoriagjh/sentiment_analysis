from django.db import models
from django.core.validators import FileExtensionValidator
from datetime import datetime
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
    def create_request(self, key, request_id, tweet_id, tweet_content, tweet_annotation, vaderScores,vaderPolarity,vaderCountpol,textblobScores,textblobPolarity,textblobCountpol, sentiWordNetScores,sentiWordNetPolarity,sentiWordNetCountpol, stanfordNLPPolarity,stanfordNLPCountpol):
        request = self.create(key=key, request_id = request_id, tweet_id=tweet_id, tweet_content=tweet_content, tweet_annotation=tweet_annotation, vaderScores=vaderScores,vaderPolarity=vaderPolarity,vaderCountpol=vaderCountpol, textblobScores=textblobScores,textblobPolarity=textblobPolarity,textblobCountpol=textblobCountpol, sentiWordNetScores=sentiWordNetScores,sentiWordNetPolarity=sentiWordNetPolarity,sentiWordNetCountpol=sentiWordNetCountpol,
        stanfordNLPPolarity = stanfordNLPPolarity,stanfordNLPCountpol=stanfordNLPCountpol)
        return request

class tweet(models.Model):
    key = models.AutoField(primary_key=True,auto_created=True)
    request_id = models.TextField(default = "something")
    tweet_id = models.TextField(default = "tweet_id")
    tweet_content = models.TextField(default = "tweet_content")
    tweet_annotation = models.TextField(default = "tweet_annotation")
    vaderScores = models.FloatField(default=150.0)
    vaderPolarity = models.CharField(max_length=200,default='SOME STRING')
    vaderCountpol = models.CharField(max_length=200,default='SOME STRING')

    textblobScores = models.FloatField(default=150.0)
    textblobPolarity = models.CharField(max_length=200,default='SOME STRING')
    textblobCountpol = models.CharField(max_length=200,default='SOME STRING')

    sentiWordNetScores = models.FloatField(default=150.0)
    sentiWordNetPolarity = models.CharField(max_length=200,default='SOME STRING')
    sentiWordNetCountpol = models.CharField(max_length=200,default='SOME STRING')

    stanfordNLPPolarity = models.CharField(max_length=200,default='SOME STRING')
    stanfordNLPCountpol = models.CharField(max_length=200,default='SOME STRING')
    stanfordNLPConfusionMatrix = models.FloatField(default=150.0)
    objects = tweetResultManager()

class requestResultManager(models.Manager):
    def create_request(self, request_id, vaderConfusionMatrix,vaderPrecise,vaderRecall,vaderF1Score, textblobConfusionMatrix,textblobPrecise,textblobRecall,textblobF1Score, sentiWordNetConfusionMatrix,sentiWordNetPrecise,sentiWordNetRecall,sentiWordNetF1Score, stanfordNLPConfusionMatrix):
        request = self.create(request_id = request_id, vaderConfusionMatrix=vaderConfusionMatrix,vaderPrecise=vaderPrecise,vaderRecall=vaderRecall,vaderF1Score=vaderF1Score, textblobConfusionMatrix=textblobConfusionMatrix,textblobPrecise=textblobPrecise,textblobRecall=textblobRecall,textblobF1Score=textblobF1Score, sentiWordNetConfusionMatrix=sentiWordNetConfusionMatrix,sentiWordNetPrecise=sentiWordNetPrecise,
        sentiWordNetRecall=sentiWordNetRecall,sentiWordNetF1Score=sentiWordNetF1Score, stanfordNLPConfusionMatrix=stanfordNLPConfusionMatrix)
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
    objects = requestResultManager()
