from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

# set the default Django settings module for the 'celery' program.
broker = 'sqla+postgresql://jooheekwon:victoriaDB@localhost/sentimentdb'

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sentiment_analysis.settings')

app = Celery(broker=broker)
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

@app.task()
def debug_task(self):
    print('Request: {0!r}'.format(self.request))
