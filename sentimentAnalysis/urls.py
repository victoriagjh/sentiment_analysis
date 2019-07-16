from django.urls import path
from sentimentAnalysis import views

urlpatterns = [
               path('', views.sentimentAnalysis, name='sentimentAnalysis'),
]
