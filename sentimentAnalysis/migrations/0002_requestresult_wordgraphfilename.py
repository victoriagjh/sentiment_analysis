# Generated by Django 2.2.6 on 2019-12-14 06:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('sentimentAnalysis', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='requestresult',
            name='wordGraphFilename',
            field=models.CharField(default='SOME STRING', max_length=200),
        ),
    ]
