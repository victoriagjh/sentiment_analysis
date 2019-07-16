from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import UploadFileForm

# Create your views here.
def sentimentAnalysis(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            #handle_uploaded_file(request.FILES['file'])
            form.save()
            return render(request, "result_page.html")
    else:
        form = UploadFileForm()
    return render(request, 'main_page.html', {'form': form})

def handle_uploaded_file(f):
    with open('text/test.txt', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
