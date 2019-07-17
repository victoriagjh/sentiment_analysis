from django.db import models
from django.core.validators import FileExtensionValidator

# Create your models here.
class UploadFileModel(models.Model):
    title = models.TextField(default='')
    file = models.FileField()
