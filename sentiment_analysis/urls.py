"""sentiment_analysis URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from sentimentAnalysis import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.main, name='main'),
    path('signIn/', views.signIn, name='signIn'),
    path('signUp/', views.signUp, name='signUp'),
    path('logout_view/', views.logout_view, name='logout_view'),
    path('history/', views.history, name = 'history'),
    path('history/(?P<request_owner>[\w.@+-]+)-(?P<request_name>[\w.@+-]+)/$', views.requestDetail, name = 'requestDetail'),
    path('history/(?P<request_owner>[\w.@+-]+)_(?P<request_name>[\w.@+-]+)/$', views.requestExplorer, name = 'requestExplorer'),
    ]
