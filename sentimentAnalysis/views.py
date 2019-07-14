from django.shortcuts import render

# Create your views here.
def sentimentAnalysis(request):
    if(request.GET.get('submitFileAndTools')):
        return render(request, 'result_page.html', {})
    return render(request, 'main_page.html', {})
