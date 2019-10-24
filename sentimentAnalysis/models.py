from django.db import models
from django.core.validators import FileExtensionValidator
from datetime import datetime
# Create your models here.
class UploadFileModel(models.Model):
    file = models.FileField(validators=[FileExtensionValidator(allowed_extensions=['txt'])])

class RequestlistManager(models.Manager):
    def create_request(self, request_id, request_owner, request_status, request_issued_time,request_completed_time,request_content):
        request= self.create(request_id = user_number, request_owner = request_owner, request_status = request_status, request_issued_time = request_issued_time,request_completed_time = request_completed_time, request_content = request_content, file_path = file_path)
        return request

class Requestlist(models.Model):
    request_id = models.TextField(default = "something", primary_key=True)
    request_owner = models.TextField(default = "something")
    request_status = models.TextField(default = "fail")
    request_issued_time = models.DateTimeField(default=datetime.now, blank=True)
    request_completed_time = models.DateTimeField(default=datetime.now, blank=True)
    request_content = models.IntegerField(default = 0)
    file_path = models.TextField(default = "/")
    objects = RequestlistManager()
