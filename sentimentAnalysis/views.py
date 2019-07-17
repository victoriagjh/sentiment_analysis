from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import UploadFileForm

# Create your views here.
def sentimentAnalysis(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            tools=[]
            if type(request.POST.get('nltk')) !=type(None):
                tools.append(request.POST.get('nltk'))
            if type(request.POST.get('spacy')) != type(None):
                tools.append(request.POST.get('spacy'))
            if type(request.POST.get('scikit')) != type(None):
                tools.append(request.POST.get('scikit'))
            if type(request.POST.get('r')) != type(None):
                tools.append(request.POST.get('r'))
            if type(request.POST.get('bagOfWords')) != type(None):
                tools.append(request.POST.get('bagOfWords'))
            if tools:
                form.save()
                form.name = request.FILES['file'].name
                content = handle_uploaded_file(request.FILES['file'])
                form.content = content
                form.tool = tools
                context = {
                    'form':form,
                    }
                return render(request, "result_page.html",context)
    else:
        form = UploadFileForm()
    context = {
        'form':form,
    }
    return render(request, 'main_page.html', context)

def handle_uploaded_file(f):
    fileName="text/"+f.name
    file = open(fileName, "r")
    content=""
    for line in file:
        content+=line
    return content
