from django.db import models

class ClientRegister_Model(models.Model):
    id = models.BigAutoField(primary_key=True)  # Explicitly define primary key
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    gender = models.CharField(max_length=30)
    address = models.CharField(max_length=30)

# Similarly for other models:
class detect_popularity_prediction(models.Model):
    id = models.BigAutoField(primary_key=True)
    photo_id = models.CharField(max_length=300)
    owner = models.CharField(max_length=300)
    gender = models.CharField(max_length=300)
    post_desc = models.CharField(max_length=300)
    score = models.CharField(max_length=300)
    created_dt = models.CharField(max_length=300)
    lat = models.CharField(max_length=300)
    lon = models.CharField(max_length=300)
    u_city = models.CharField(max_length=300)
    u_country = models.CharField(max_length=300)
    Prediction = models.CharField(max_length=300)

class detection_accuracy(models.Model):
    id = models.BigAutoField(primary_key=True)
    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):
    id = models.BigAutoField(primary_key=True)
    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)