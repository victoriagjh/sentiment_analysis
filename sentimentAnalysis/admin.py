from django.contrib import admin
from .models import Request,requestResult,tweet,sentenceResult,tasklist

# Register your models here.
admin.site.register(Request)
admin.site.register(requestResult)
admin.site.register(tweet)
admin.site.register(sentenceResult)
admin.site.register(tasklist)
