from __future__ import absolute_import, unicode_literals
from celery import shared_task
from .models import Requestlist
import random,os,re,time

@shared_task
def run(id):
    print("Process ID : " + str(os.getpid()))
    request = Requestlist.objects.get(request_id=id)
    request.request_status = "pending"
    request.save()
    filePath = request.file_path
    file = open(filePath, "r", encoding='utf-8',errors="ignore")
    text=file.readlines()
    id = []
    content = []
    hashtag = []
    annotation = []

    pattern = '#([0-9a-zA-Z]*)'
    hashtag_word = re.compile(pattern)

    for line in text:
        sentence = re.split(r'\t+', line)
        text = ""
        id.append(sentence[0])
        content.append(sentence[1])
        annotation.append(sentence[2].strip('\n'))

        for tag in hashtag_word.findall(line):
            hashtag.append(tag)
    #check

    request.request_status = "success"
    request.request_completed_time = time.strftime(r"%Y-%m-%d %H:%M:%S", time.localtime())
    request.save()
    print("SUCCESS : [ "+str(id)+" ] ")
