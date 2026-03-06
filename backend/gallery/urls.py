from django.urls import path
from . import views

urlpatterns = [
    path('', views.gallery_view, name='gallery'),
    path('upload/', views.upload_photo, name='upload_photo'),
    path('rescan/', views.rescan_view,   name='rescan'),
    path('people/',  views.people_view,   name='people_view'),
    path('search/',  views.visual_search, name='visual_search'),
    path('search/', views.visual_search, name='visual_search'),
    path('nl-search/', views.nl_search,   name='nl_search'), 
    path('batch-upload/', views.batch_upload, name='batch_upload'),
]