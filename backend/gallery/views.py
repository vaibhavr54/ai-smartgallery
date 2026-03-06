# gallery/views.py

from django.shortcuts import render, redirect
from .models import Photo, Face, Embedding, Person
from .forms import PhotoUploadForm
from ai_engine.face_recognition.detector import detect_faces
from ai_engine.face_recognition.matcher import find_matching_person, update_average_embedding
from ai_engine.face_recognition.clusterer import run_dbscan_clustering


# ---------- FACE QUALITY FILTERS ----------
MIN_FACE_SIZE  = 80     # pixels
MIN_CONFIDENCE = 0.60   # detector confidence
MIN_RATIO      = 0.6    # width/height ratio
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
        "person_count":  Person.objects.count(),   # ← add this
        "event_count":   Event.objects.count(),    # ← add this
    })

def rescan_view(request):
    """Triggers a full DBSCAN recluster of all faces in the library."""
    result = run_dbscan_clustering()
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