from django.contrib import admin
from .models import requestResult, requestResultManager
from .models import tweet, tweetResultManager
from .models import sentenceResultManager, sentenceResult

# Register your models here.
admin.site.register(requestResult)
admin.site.register(tweet)
admin.site.register(sentenceResult)
