from django.shortcuts import render

# Create your views here.
def sentimentAnalysis(request):
    return render(request, 'main_page.html', {})
