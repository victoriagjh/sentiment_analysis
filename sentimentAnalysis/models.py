from django.db import models
from django.core.validators import FileExtensionValidator

# Create your models here.
class UploadFileModel(models.Model):
    file = models.FileField(validators=[FileExtensionValidator(allowed_extensions=['txt'])])

class RequestlistManager(models.Manager):
    def create_request(self, user_number, project_number, request_time, request_status):
        request= self.create(user_number = user_number, project_number = project_number, project_status = request_status)
        return request    

class Requestlist(models.Model):
    user_id = models.TextField(default = "something", primary_key=True)
    project_number = models.IntegerField(default = 0)
    project_status = models.TextField(default = "fail")

    objects = RequestlistManager()