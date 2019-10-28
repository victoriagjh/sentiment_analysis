from django.contrib import admin
from .models import Requestlist,requestResult,tweet

# Register your models here.
admin.site.register(Requestlist)
admin.site.register(requestResult)
admin.site.register(tweet)
