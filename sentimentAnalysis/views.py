from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import UploadFileForm

# Create your views here.
def sentimentAnalysis(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            content = handle_uploaded_file(request.FILES['file'])
            form.save()
            form.name = request.FILES['file'].name
            form.content = content
            context = {
                'form':form,
            }
            return render(request, "result_page.html",context)
    else:
        form = UploadFileForm()
    return render(request, 'main_page.html', {'form': form})

def handle_uploaded_file(f):
    fileName="text/"+f.name
    file = open(fileName, "r")
    content=""
    for line in file:
        content+=line
    return content
