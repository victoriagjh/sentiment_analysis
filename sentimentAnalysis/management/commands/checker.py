from django.core.management.base import BaseCommand
#from myapp.models import MyModel
import threading
import time

def execute(number):
    print(threading.current_thread().getName(), number)

class Command(BaseCommand):
    help = 'check the latest instance of MyModel'
    def handle(self, *args, **kwargs):
        cnt = 0
        while True:
            time.sleep(0.5)
            my_thread = threading.Thread(target=execute, args=(cnt,))
            my_thread.start()
            cnt += 1
