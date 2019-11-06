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
    path('expert/', views.expert_page, name='expert_page'),
    path('', views.sentimentAnalysis, name='sentimentAnalysis'),
    path('signIn/', views.signIn, name='signIn'),
    path('logout_view/', views.logout_view, name='logout_view'),
    path('postsign/', views.postsign, name="post"),
    path('logout/', views.logout, name='log'),
    path('signUp/', views.postsign, name='signup'),
    path('postsignup/', views.postsignup, name='signup'),
    path('accounts/profile/', views.afterlogin, name='afterlogin'),
    path('auth/', include('social_django.urls', namespace='social')),
]
