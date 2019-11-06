from django.db import models
from django.core.validators import FileExtensionValidator
from django.contrib.auth.base_user import AbstractBaseUser, BaseUserManager

# Create your models here.
class UploadFileModel(models.Model):
    file = models.FileField(validators=[FileExtensionValidator(allowed_extensions=['txt'])])


class SAResultSentenceManager(models.Manager):
    def createSentenceresult(self, ids, vaderSentenceScoreList, vaderSentencePolarityList, vaderAverage, vaderMajority, 
                               textblobSentenceScoreList, textblobSentencePolarityList, textblobAverage, textblobMajority, 
                               stanfordNLPSentencePolarityList, stanfordNLPMajority, 
                               sentiWordNetSentenceScoreList, sentiWordNetSentencePolarityList, sentiWordNetAverage, sentiWordNetMajority, 
                               sentenceKappa):
        res = self.create(ids = ids, vaderSentenceScoreList = vaderSentenceScoreList, vaderSentencePolarityList = vaderSentencePolarityList, vaderAverage = vaderAverage, vaderMajority =vaderMajority, 
        textblobSentenceScoreList = textblobSentenceScoreList, textblobSentencePolarityList = textblobSentencePolarityList, textblobAverage = textblobAverage, textblobMajority = textblobMajority, 
        stanfordNLPSentencePolarityList = stanfordNLPSentencePolarityList, stanfordNLPMajority = stanfordNLPMajority, 
        sentiWordNetSentenceScoreList = sentiWordNetSentenceScoreList, sentiWordNetSentencePolarityList = sentiWordNetSentencePolarityList, sentiWordNetAverage = sentiWordNetAverage, sentiWordNetMajority = sentiWordNetMajority, 
        sentenceKappa = sentenceKappa)
        return res

class SAResultSentence(models.Model):
    ids = models.CharField(max_length = 200, primary_key=True, default = "SOME STRING")
    vaderSentenceScoreList = models.CharField(max_length = 200, default = "SOME STRING")
    vaderSentencePolarityList = models.CharField(max_length = 200, default = "SOME STRING")
    vaderAverage = models.FloatField(default=150.0)
    vaderMajority = models.CharField(max_length = 200, default = "SOME STRING")
    textblobSentenceScoreList = models.CharField(max_length = 200, default = "SOME STRING")
    textblobSentencePolarityList = models.CharField(max_length = 200, default = "SOME STRING")
    textblobAverage = models.FloatField(default=150.0)
    textblobMajority = models.CharField(max_length = 200, default = "SOME STRING")
    stanfordNLPSentencePolarityList = models.CharField(max_length = 200, default = "SOME STRING")
    stanfordNLPMajority = models.CharField(max_length = 200, default = "SOME STRING")
    sentiWordNetSentenceScoreList = models.CharField(max_length = 200, default = "SOME STRING")
    sentiWordNetSentencePolarityList = models.CharField(max_length = 200, default = "SOME STRING")
    sentiWordNetAverage = models.FloatField(default=150.0)
    sentiWordNetMajority = models.CharField(max_length = 200, default = "SOME STRING")
    sentenceKappa = models.FloatField(default=150.00, null=True)
    objects = SAResultSentenceManager()
    
class SAResultManager(models.Manager):
    def create_result(self,ids,tweetContent,vaderTweetScore,vaderTweetPolarity,textblobTweetScore,textblobTweetPolarity,
    stanfordNLPTweetPolarity,sentiWordNetTweetScore,sentiWordNetTweetPolarity,tweetKappa):
        res=self.create(ids=ids,tweetContent=tweetContent,vaderTweetScore=vaderTweetScore,vaderTweetPolarity=vaderTweetPolarity,
        textblobTweetScore=textblobTweetScore,textblobTweetPolarity=textblobTweetPolarity,
        stanfordNLPTweetPolarity=stanfordNLPTweetPolarity,sentiWordNetTweetScore=sentiWordNetTweetScore,sentiWordNetTweetPolarity=sentiWordNetTweetPolarity,tweetKappa=tweetKappa)
        return res

class SAResult(models.Model):
    ids=models.CharField(max_length=200,primary_key=True,default="SOME STRING")
    tweetContent=models.CharField(max_length=200,default='SOME STRING')
    vaderTweetScore=models.FloatField(default=150.0)
    vaderTweetPolarity=models.CharField(max_length=200,default='SOME STRING')
    textblobTweetScore=models.FloatField(default=150.00)
    textblobTweetPolarity=models.CharField(max_length=200,default='SOME STRING')
    stanfordNLPTweetPolarity=models.CharField(max_length=200,default='SOME STRING')
    sentiWordNetTweetScore=models.FloatField(default=150.00)
    sentiWordNetTweetPolarity=models.CharField(max_length=200,default='SOME STRING')
    tweetKappa=models.FloatField(default=150.00,null=True)
    
    objects = SAResultManager()
