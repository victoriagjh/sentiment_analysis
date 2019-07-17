from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import UploadFileForm
from .forms import ToolsForm

# Create your views here.
def sentimentAnalysis(request):
    toolForm = ToolsForm()
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            form.name = request.FILES['file'].name
            content = handle_uploaded_file(request.FILES['file'])
            form.content = content
            context = {
                'form':form,
            }
            return render(request, "result_page.html",context)
    else:
        form = UploadFileForm()
    context = {
        'form':form,
        'toolForm':toolForm
    }
    return render(request, 'main_page.html', context)

def handle_uploaded_file(f):
    fileName="text/"+f.name
    file = open(fileName, "r")
    content=""
    for line in file:
        content+=line
    return content
