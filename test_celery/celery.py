from __future__ import absolute_import
from celery import Celery

app = Celery('test_celery', 
            broker = 'amqp://easy:easy123@localhost/test_vhost',
            backend='rpc://', 
            include=['test_celery.tasks'])