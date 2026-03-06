from django.contrib import admin
from .models import Photo, Face, Person, Event

admin.site.register(Photo)
admin.site.register(Face)
admin.site.register(Person)
admin.site.register(Event)
