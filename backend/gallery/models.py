from django.db import models


class Event(models.Model):
    name = models.CharField(max_length=255)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()

    def __str__(self):
        return self.name


class Person(models.Model):
    label        = models.CharField(max_length=255, blank=True)
    created_at   = models.DateTimeField(auto_now_add=True)

    # --- ADD THESE TWO ---
    avg_embedding = models.JSONField(null=True, blank=True)  # running average of all face embeddings
    face_count    = models.IntegerField(default=0)           # how many faces contributed to the average

    def __str__(self):
        return self.label or f"Person {self.id}"


class Photo(models.Model):
    image           = models.ImageField(upload_to='photos/')
    uploaded_at     = models.DateTimeField(auto_now_add=True)
    event           = models.ForeignKey(Event, on_delete=models.SET_NULL, null=True, blank=True)
    clip_embedding  = models.JSONField(null=True, blank=True)  # ← only new line

    def __str__(self):
        return f"Photo {self.id}"


class Face(models.Model):
    photo = models.ForeignKey(Photo, on_delete=models.CASCADE, related_name='faces')
    person = models.ForeignKey(Person, on_delete=models.SET_NULL, null=True, blank=True)

    # Bounding box
    x = models.IntegerField()
    y = models.IntegerField()
    width = models.IntegerField()
    height = models.IntegerField()

    # Detection confidence (useful later)
    confidence = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"Face {self.id} in Photo {self.photo.id}"


class Embedding(models.Model):
    face = models.OneToOneField(Face, on_delete=models.CASCADE, related_name='embedding')
    vector = models.JSONField()  # 512-D face embedding

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Embedding for Face {self.face.id}"
