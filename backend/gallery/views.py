# gallery/views.py

from django.shortcuts import render, redirect
from .models import Photo, Face, Embedding, Person
from .forms import PhotoUploadForm
from ai_engine.face_recognition.detector import detect_faces
from ai_engine.face_recognition.matcher import find_matching_person, update_average_embedding
from ai_engine.face_recognition.clusterer import run_dbscan_clustering
from django.http import JsonResponse
from django.views.decorators.http import require_POST
import json


# ---------- FACE QUALITY FILTERS ----------
MIN_FACE_SIZE  = 50     # pixels
MIN_CONFIDENCE = 0.45   # detector confidence
MIN_RATIO      = 0.5    # width/height ratio
MAX_RATIO      = 1.6


def valid_face(width, height, confidence):
    if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
        return False
    if confidence is not None and confidence < MIN_CONFIDENCE:
        return False
    ratio = width / float(height)
    if ratio < MIN_RATIO or ratio > MAX_RATIO:
        return False
    return True


# ---------- CORE PIPELINE ----------
def process_photo(photo):
    """
    Runs the full AI pipeline on a saved Photo object:
      1. Detect faces
      2. Quality filter
      3. Save Face + Embedding
      4. Match to existing Person or create new one
      5. Update Person's average embedding
    """
    image_path = photo.image.path
    print(f"\n=== PIPELINE START | Photo {photo.id} ===")

    try:
        detections = detect_faces(image_path) or []
    except Exception as e:
        print(f"[pipeline] Detection error: {e}")
        return

    print(f"[pipeline] Raw detections: {len(detections)}")

    # Load all persons that already have an average embedding
    known_persons = list(Person.objects.filter(avg_embedding__isnull=False))

    saved = 0
    for d in detections:

        # --- Extract fields ---
        x         = d.get("x")
        y         = d.get("y")
        width     = d.get("width")
        height    = d.get("height")
        confidence = d.get("det_score")
        embedding  = d.get("embedding")

        if None in (x, y, width, height):
            continue

        # --- Quality filter ---
        if not valid_face(width, height, confidence):
            print(f"[pipeline] Face rejected (w={width} h={height} conf={confidence:.2f})")
            continue

        # --- Save Face ---
        face_obj = Face.objects.create(
            photo      = photo,
            x          = x,
            y          = y,
            width      = width,
            height     = height,
            confidence = float(confidence) if confidence else None,
        )

        # --- Save Embedding ---
        if embedding is not None:
            Embedding.objects.create(face=face_obj, vector=embedding)

            # --- Person Matching ---
            matched_person = find_matching_person(embedding, known_persons)

            if matched_person:
                face_obj.person = matched_person
                face_obj.save()
                update_average_embedding(matched_person, embedding)

            else:
                # Create a brand new Person for this unrecognised face
                new_person = Person.objects.create()
                face_obj.person = new_person
                face_obj.save()
                update_average_embedding(new_person, embedding)

                # Add to known list so subsequent faces in THIS photo can match it
                known_persons.append(new_person)

        saved += 1
        print(f"[pipeline] Saved Face {face_obj.id} → Person {face_obj.person_id}")
        # --- Generate CLIP embedding for the whole photo ---
        try:
            from ai_engine.clip_search.encoder import encode_image
            photo.clip_embedding = encode_image(image_path)
            photo.save()
            print(f"[pipeline] CLIP embedding saved for Photo {photo.id}")
        except Exception as e:
            print(f"[pipeline] CLIP embedding failed: {e}")
        # --- Assign Event based on timestamp ---
        try:
            from ai_engine.event_grouping.grouper import assign_event_to_photo
            assign_event_to_photo(photo)
            print(f"[pipeline] Event assigned for Photo {photo.id}")
        except Exception as e:
            print(f"[pipeline] Event assignment failed: {e}")

    print(f"=== PIPELINE END | {saved} faces saved ===\n")


# ---------- UPLOAD VIEW ----------
def upload_photo(request):
    if request.method == 'POST':
        form = PhotoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            photo = form.save()
            process_photo(photo)
            return redirect('gallery')
    else:
        form = PhotoUploadForm()

    return render(request, 'gallery/upload.html', {'form': form})


from .models import Photo, Face, Embedding, Person, Event

def gallery_view(request):
    photos = Photo.objects.all().order_by('-uploaded_at')
    photo_faces = []
    for photo in photos:
        photo_faces.append({
            "photo": photo,
            "faces": photo.faces.select_related('person').all()
        })
    return render(request, "gallery/gallery.html", {
        "photo_faces":   photo_faces,
        "person_count":  Person.objects.count(),
        "event_count":   Event.objects.count(),    
        "face_count":    Face.objects.count(),
    })

def rescan_view(request):
    result = run_dbscan_clustering()

    # Also regroup all events
    try:
        from ai_engine.event_grouping.grouper import group_photos_into_events
        event_result = group_photos_into_events()
        result['events_created'] = event_result['events_created']
        result['photos_grouped'] = event_result['photos_grouped']
    except Exception as e:
        result['event_error'] = str(e)

    return render(request, 'gallery/rescan.html', {'result': result})

def people_view(request):
    persons = Person.objects.all()
    return render(request, 'gallery/people.html', {
        'persons': persons,
        'person_count': persons.count(),
        'event_count': Event.objects.count(),
    })

def visual_search(request):
    results   = []
    query_img = None
    error     = None

    if request.method == 'POST' and request.FILES.get('query_image'):
        import tempfile, os
        from ai_engine.face_recognition.matcher import cosine_similarity
        from .models import Embedding

        query_file = request.FILES['query_image']

        # Save to a temp file so InsightFace can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            for chunk in query_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            detections = detect_faces(tmp_path) or []
        except Exception as e:
            error = f"Detection failed: {e}"
            detections = []
        finally:
            os.unlink(tmp_path)  # always clean up temp file

        if not detections:
            error = "No face detected in the uploaded image. Try a clearer photo."
        else:
            # Use the highest-confidence face from the query image
            best = max(detections, key=lambda d: d.get('det_score') or 0)
            query_vec = best.get('embedding')

            if query_vec is None:
                error = "Could not extract face embedding. Try a different photo."
            else:
                # Compare against every stored embedding
                SEARCH_THRESHOLD = 0.45
                seen_photos = {}  # photo_id → best score (deduplicate)

                for emb in Embedding.objects.select_related('face__photo', 'face__person').all():
                    if emb.vector is None:
                        continue
                    score = cosine_similarity(query_vec, emb.vector)
                    if score >= SEARCH_THRESHOLD:
                        photo = emb.face.photo
                        if photo.id not in seen_photos or score > seen_photos[photo.id]['score']:
                            seen_photos[photo.id] = {
                                'photo':  photo,
                                'person': emb.face.person,
                                'score':  score,
                                'face':   emb.face,
                            }

                # Sort by score descending
                results = sorted(seen_photos.values(), key=lambda x: x['score'], reverse=True)
                query_img = query_file.name

    return render(request, 'gallery/visual_search.html', {
        'results':      results,
        'query_img':    query_img,
        'error':        error,
        'person_count': Person.objects.count(),
        'event_count':  Event.objects.count(),
        'searched':     request.method == 'POST',
    })

def nl_search(request):
    from ai_engine.clip_search.encoder import encode_text
    from ai_engine.face_recognition.matcher import cosine_similarity

    results = []
    query   = ''
    error   = None
    searched = False

    if request.method == 'GET' and request.GET.get('q'):
        query    = request.GET.get('q', '').strip()
        searched = True

        if not query:
            error = "Please enter a search query."
        else:
            try:
                query_vec = encode_text(query)
                THRESHOLD = 0.20  # CLIP similarity is lower than face similarity

                scored = []
                for photo in Photo.objects.exclude(clip_embedding__isnull=True):
                    score = cosine_similarity(query_vec, photo.clip_embedding)
                    if score >= THRESHOLD:
                        scored.append({'photo': photo, 'score': score})

                results = sorted(scored, key=lambda x: x['score'], reverse=True)[:20]

            except Exception as e:
                error = f"Search failed: {e}"

    return render(request, 'gallery/nl_search.html', {
        'results':       results,
        'query':         query,
        'error':         error,
        'searched':      searched,
        'person_count':  Person.objects.count(),
        'event_count':   Event.objects.count(),
    })

def batch_upload(request):
    if request.method == 'POST':
        images = request.FILES.getlist('images')  # getlist handles multiple files
        
        if not images:
            return render(request, 'gallery/batch_upload.html', {
                'error': 'No images selected.',
                'person_count': Person.objects.count(),
                'event_count': Event.objects.count(),
            })

        results = []
        for image_file in images:
            # Create Photo object manually (not via form)
            photo = Photo.objects.create()
            photo.image.save(image_file.name, image_file, save=True)
            
            # Run full AI pipeline
            try:
                process_photo(photo)
                # Generate CLIP embedding
                from ai_engine.clip_search.encoder import encode_image
                photo.clip_embedding = encode_image(photo.image.path)
                photo.save()
                results.append({'name': image_file.name, 'status': 'ok', 'id': photo.id})
            except Exception as e:
                results.append({'name': image_file.name, 'status': 'error', 'error': str(e)})

        return render(request, 'gallery/batch_upload.html', {
            'results':      results,
            'done':         True,
            'person_count': Person.objects.count(),
            'event_count':  Event.objects.count(),
        })

    return render(request, 'gallery/batch_upload.html', {
        'person_count': Person.objects.count(),
        'event_count':  Event.objects.count(),
    })

def people_view(request):
    persons = Person.objects.prefetch_related(
        'face_set__photo',
        'face_set__embedding'
    ).all().order_by('-face_count')

    people_data = []
    for person in persons:
        # Get best face crop — highest confidence face
        best_face = person.face_set.filter(
            confidence__isnull=False
        ).order_by('-confidence').first()

        if not best_face:
            best_face = person.face_set.first()

        # Get all photos this person appears in
        photo_ids = person.face_set.values_list(
            'photo_id', flat=True
        ).distinct()
        photo_count = photo_ids.count()

        people_data.append({
            'person':      person,
            'best_face':   best_face,
            'photo_count': photo_count,
        })

    return render(request, 'gallery/people.html', {
        'people_data':  people_data,
        'person_count': persons.count(),
        'event_count':  Event.objects.count(),
        'face_count':   Face.objects.count(),
    })


def person_detail(request, person_id):
    from django.shortcuts import get_object_or_404
    person = get_object_or_404(Person, id=person_id)

    # Rename via POST
    if request.method == 'POST':
        new_label = request.POST.get('label', '').strip()
        if new_label:
            person.label = new_label
            person.save()
        return redirect('person_detail', person_id=person.id)

    # All photos this person appears in
    faces = person.face_set.select_related('photo').all()
    photo_ids = faces.values_list('photo_id', flat=True).distinct()
    photos = Photo.objects.filter(id__in=photo_ids).order_by('-uploaded_at')

    # Best face for header
    best_face = person.face_set.filter(
        confidence__isnull=False
    ).order_by('-confidence').first() or person.face_set.first()

    return render(request, 'gallery/person_detail.html', {
        'person':       person,
        'photos':       photos,
        'best_face':    best_face,
        'face_count':   faces.count(),
        'person_count': Person.objects.count(),
        'event_count':  Event.objects.count(),
    })


@require_POST
def rename_person(request, person_id):
    from django.shortcuts import get_object_or_404
    person = get_object_or_404(Person, id=person_id)
    try:
        data = json.loads(request.body)
        new_label = data.get('label', '').strip()
        if new_label:
            person.label = new_label
            person.save()
            return JsonResponse({'status': 'ok', 'label': person.label})
        return JsonResponse({'status': 'error', 'msg': 'Empty label'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'msg': str(e)}, status=500)
    
def face_crop(request, face_id):
    """Returns a cropped face thumbnail as JPEG response."""
    from django.shortcuts import get_object_or_404
    from django.http import HttpResponse
    from PIL import Image
    import io

    face = get_object_or_404(Face, id=face_id)
    img = Image.open(face.photo.image.path).convert('RGB')

    # Add padding around the face bbox
    pad = 30
    x1 = max(0, face.x - pad)
    y1 = max(0, face.y - pad)
    x2 = min(img.width,  face.x + face.width  + pad)
    y2 = min(img.height, face.y + face.height + pad)

    crop = img.crop((x1, y1, x2, y2))
    crop = crop.resize((200, 200), Image.LANCZOS)

    buf = io.BytesIO()
    crop.save(buf, format='JPEG', quality=90)
    buf.seek(0)
    return HttpResponse(buf, content_type='image/jpeg')

def events_view(request):
    from ai_engine.event_grouping.grouper import group_photos_into_events

    # Trigger regrouping if requested
    if request.GET.get('regroup') == '1':
        group_photos_into_events()
        return redirect('events_view')

    events = Event.objects.all().order_by('-start_time')
    events_data = []

    for event in events:
        photos = Photo.objects.filter(event=event).order_by('uploaded_at')
        events_data.append({
            'event':       event,
            'photos':      photos,
            'cover':       photos.first(),
            'photo_count': photos.count(),
        })

    return render(request, 'gallery/events.html', {
        'events_data':  events_data,
        'event_count':  events.count(),
        'person_count': Person.objects.count(),
    })

def enhance_view(request):
    enhanced_url = None
    error        = None
    original_url = None
    photo        = None
    processing   = False

    # Enhance from gallery photo
    photo_id = request.GET.get('photo_id') or request.POST.get('photo_id')
    if photo_id:
        try:
            photo = Photo.objects.get(id=photo_id)
            original_url = photo.image.url
        except Photo.DoesNotExist:
            error = "Photo not found."

    if request.method == 'POST' and not error:
        import uuid
        from ai_engine.enhancer.enhancer import enhance_image

        try:
            processing = True

            # Get input path
            if photo:
                input_path = photo.image.path
                out_name   = f"enhanced_{photo.id}_{uuid.uuid4().hex[:6]}.jpg"
            elif request.FILES.get('upload'):
                import tempfile
                f = request.FILES['upload']
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    for chunk in f.chunks():
                        tmp.write(chunk)
                    input_path = tmp.name
                original_url = None
                out_name = f"enhanced_upload_{uuid.uuid4().hex[:8]}.jpg"
            else:
                raise ValueError("No image provided.")

            output_path  = enhance_image(input_path, out_name)
            enhanced_url = '/' + output_path.replace('\\', '/')
            processing   = False

        except Exception as e:
            error      = f"Enhancement failed: {e}"
            processing = False

    # Gallery photos for picker
    gallery_photos = Photo.objects.all().order_by('-uploaded_at')[:40]

    return render(request, 'gallery/enhance.html', {
        'photo':          photo,
        'original_url':   original_url,
        'enhanced_url':   enhanced_url,
        'error':          error,
        'processing':     processing,
        'gallery_photos': gallery_photos,
        'person_count':   Person.objects.count(),
        'event_count':    Event.objects.count(),
    })


def save_enhanced_to_gallery(request):
    """Save an enhanced photo back to the gallery as a new Photo object."""
    if request.method == 'POST':
        import shutil, uuid
        enhanced_path = request.POST.get('enhanced_path', '').lstrip('/')

        if not enhanced_path or not os.path.exists(enhanced_path):
            return JsonResponse({'status': 'error', 'msg': 'File not found'}, status=400)

        # Copy into photos/ media folder
        ext      = os.path.splitext(enhanced_path)[1]
        new_name = f"photos/enhanced_{uuid.uuid4().hex[:10]}{ext}"
        dest     = os.path.join('media', new_name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(enhanced_path, dest)

        # Create Photo object and run pipeline
        new_photo = Photo.objects.create()
        new_photo.image.name = new_name
        new_photo.save()
        process_photo(new_photo)

        return JsonResponse({'status': 'ok', 'photo_id': new_photo.id})

    return JsonResponse({'status': 'error'}, status=405)

def collage_view(request):
    result_url   = None
    error        = None
    selected_ids = []

    if request.method == 'POST':
        import uuid
        from ai_engine.collage.collage_engine import build_collage

        selected_ids = request.POST.getlist('photo_ids')

        if len(selected_ids) < 2:
            error = "Please select at least 2 photos."
        elif len(selected_ids) > 12:
            error = "Maximum 12 photos per collage."
        else:
            try:
                photos_with_faces = []
                for pid in selected_ids:
                    photo = Photo.objects.get(id=pid)
                    faces = list(photo.faces.all())
                    photos_with_faces.append((photo, faces))

                out_name   = f"collage_{uuid.uuid4().hex[:10]}.jpg"
                output_path = build_collage(photos_with_faces, out_name)
                result_url  = '/' + output_path.replace('\\', '/')

                # Save to gallery if requested
                if request.POST.get('save_to_gallery'):
                    import shutil
                    new_name = f"photos/collage_{uuid.uuid4().hex[:8]}.jpg"
                    dest = os.path.join('media', new_name)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copy2(output_path, dest)
                    new_photo = Photo.objects.create()
                    new_photo.image.name = new_name
                    new_photo.save()

            except Exception as e:
                error = f"Collage generation failed: {e}"

    gallery_photos = Photo.objects.all().order_by('-uploaded_at')

    return render(request, 'gallery/collage.html', {
        'gallery_photos': gallery_photos,
        'result_url':     result_url,
        'error':          error,
        'selected_ids':   [int(i) for i in selected_ids],
        'person_count':   Person.objects.count(),
        'event_count':    Event.objects.count(),
    })

import os
from django.http import JsonResponse
from django.views.decorators.http import require_POST

@require_POST
def delete_photo(request, photo_id):
    from django.shortcuts import get_object_or_404
    photo = get_object_or_404(Photo, id=photo_id)
    try:
        if os.path.exists(photo.image.path):
            os.remove(photo.image.path)
    except Exception:
        pass
    photo.delete()
    return JsonResponse({'status': 'ok'})


@require_POST
def bulk_delete_photos(request):
    import json
    try:
        data = json.loads(request.body)
        ids  = data.get('ids', [])
    except Exception:
        return JsonResponse({'status': 'error', 'msg': 'Invalid JSON'}, status=400)

    deleted = 0
    for pid in ids:
        try:
            photo = Photo.objects.get(id=pid)
            try:
                if os.path.exists(photo.image.path):
                    os.remove(photo.image.path)
            except Exception:
                pass
            photo.delete()
            deleted += 1
        except Photo.DoesNotExist:
            pass

    return JsonResponse({'status': 'ok', 'deleted': deleted})

def stats_view(request):
    return JsonResponse({
        'face_count':   Face.objects.count(),
        'person_count': Person.objects.count(),
        'event_count':  Event.objects.count(),
    })