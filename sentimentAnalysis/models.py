from django.db import models
from django.core.validators import FileExtensionValidator

# Create your models here.
class UploadFileModel(models.Model):
    title = models.TextField(default='')
    file = models.FileField()

class ToolModel(models.Model):
    tool1 = models.BooleanField(default=False)
    tool2 = models.BooleanField(default=False)
    tool3 = models.BooleanField(default=False)
    tool4 = models.BooleanField(default=False)
    tool5 = models.BooleanField(default=False)
