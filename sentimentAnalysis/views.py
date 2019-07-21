from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import UploadFileForm
from django.contrib import messages

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
                text = handle_uploaded_file(request.FILES['file'])
                content = getContent(text)
                form.text = text
                form.content = content
                form.tool = tools
                context = {
                    'form':form,
                    }
                return render(request, "result_page.html",context)
            if not tools:
                messages.warning(request, 'You should check the tool at least 1!', extra_tags='alert')
    else:
        form = UploadFileForm()
    context = {
        'form':form,
    }
    return render(request, 'main_page.html', context)

def result_page(request):
    if request.method == 'POST':
        if 'spacy_button' in request.POST:
            context = {
            'spacy':"spacy",
            }
            return render(request, 'last.html', context)
    return render(request,'main_page.html',context)

def handle_uploaded_file(f):
    fileName="text/"+f.name
    file = open(fileName, "r")
    text=file.readlines()
    textList=[]
    for i in text:
        i=i.replace('\n','\t')
        i=i.replace('   ','\t')
        i=i.split('\t')
        for j in i:
            if j!='':
                textList.append(j)
    file.close()
    return textList

def getContent(text):
    content=[]
    s=0
    for i in text:
        if s%3==1 and s!=1:
            content.append(i)
        s+=1
    return content
