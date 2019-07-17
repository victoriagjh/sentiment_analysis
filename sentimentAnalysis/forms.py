from django import forms

from .models import UploadFileModel
from .models import ToolModel

class UploadFileForm(forms.ModelForm):
    class Meta:
        model = UploadFileModel
        fields = ('title', 'file')
    def __init__(self, *args, **kwargs):
        super(UploadFileForm, self).__init__(*args, **kwargs)
        self.fields['file'].required = False

class ToolsForm(forms.ModelForm):
    class Meta:
        model = ToolModel
        fields = ["tool1","tool2","tool3","tool4","tool5"]
        labels = {"tool1": "NLTK","tool2":"Spacy","tool3":"Scikit","tool4":"R","tool5":"Bag of Words"}
