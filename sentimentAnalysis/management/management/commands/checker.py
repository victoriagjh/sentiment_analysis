from django.core.management.base import BaseCommand
from sentimentAnalysis.models import Request,tweet,requestResult,sentenceResult,tasklist
import threading
import time
from sentimentAnalysis.tasks import totalAnalysis
import ast

def execute(number):
    cnt = number
    while True:
        requestList = Request.objects.filter(request_status = "pending").values()
        for request in requestList :
            print("request : ", request)
            vader = tasklist.objects.filter(request_key = request['key'], toolName = "vader").first()
            text = tasklist.objects.filter(request_key = request['key'], toolName ="textblob").first()
            senti = tasklist.objects.filter(request_key = request['key'], toolName ="sentiWordNet").first()
            stan = tasklist.objects.filter(request_key = request['key'], toolName ="stanfordNLP").first()

            if vader.toolStatus == "success" and vader.isMailSended == "NOT YET" and text.toolStatus == "success" and text.isMailSended == "NOT YET" and senti.toolStatus =="success" and senti.isMailSended == "NOT YET" and stan.toolStatus == "success" and stan.isMailSended == "NOT YET":
                req = requestResult.objects.filter(requestName = request['request_name'], userEmail = request['request_owner']).first()
                tweetIDs= ast.literal_eval(req.tweetIDs)
                totalAnalysis.apply_async(kwargs={'requestName': request['request_name'],'email': request['request_owner'],'tweet_id': tweetIDs},time_limit=60*30, soft_time_limit=60*30)
        time.sleep(10) #interval is 10 minites

class Command(BaseCommand):
    help = 'check the latest instance of MyModel'
    def handle(self, *args, **kwargs):
        my_thread = threading.Thread(target=execute, args=(0,))
        my_thread.start()
