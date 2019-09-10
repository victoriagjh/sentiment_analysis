from django.db import models
from django.core.validators import FileExtensionValidator

# Create your models here.
class UploadFileModel(models.Model):
    file = models.FileField(validators=[FileExtensionValidator(allowed_extensions=['txt'])])

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
