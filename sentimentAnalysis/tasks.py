from __future__ import absolute_import, unicode_literals
from celery import shared_task
from .models import Requestlist
import time
import random
import os

@shared_task
def run(id):
    print("Process ID : " + str(os.getpid()))
    request = Requestlist.objects.get(request_id=id)
    request.request_status = "pending"
    request.save()
    #sleepTime=random.randrange(0,100)
    sleepTime=45
    time.sleep(sleepTime)
    request.request_status = "success"
    request.request_completed_time = time.strftime(r"%Y-%m-%d %H:%M:%S", time.localtime())
    request.save()
    print("SUCCESS : [ "+str(id)+" ] sleep time - "+str(sleepTime))
