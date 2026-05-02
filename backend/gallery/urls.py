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
    path('people/', views.people_view,    name='people_view'),
    path('people/<int:person_id>/',  views.person_detail,  name='person_detail'),
    path('people/<int:person_id>/rename/', views.rename_person, name='rename_person'),
    path('face-crop/<int:face_id>/', views.face_crop, name='face_crop'),
    path('events/', views.events_view, name='events_view'),
    path('enhance/', views.enhance_view, name='enhance'),
    path('enhance/save/', views.save_enhanced_to_gallery, name='save_enhanced'),
    path('collage/', views.collage_view, name='collage'),
    path('photo/<int:photo_id>/delete/', views.delete_photo,      name='delete_photo'),
    path('photos/bulk-delete/',          views.bulk_delete_photos, name='bulk_delete'),
    path('stats/', views.stats_view, name='stats'),
]