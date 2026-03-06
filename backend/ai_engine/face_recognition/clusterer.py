# ai_engine/face_recognition/clusterer.py

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from gallery.models import Embedding, Face, Person


# --- Tuning knobs ---
# eps: max cosine distance between two faces to be considered neighbours
#      lower = stricter clusters (fewer false merges)
#      higher = looser clusters (fewer missed matches)
EPS          = 0.60

# min_samples: minimum faces needed to form a cluster (not noise)
#              1 = every single face becomes its own person (no orphans)
#              2 = a person must appear at least twice to get a cluster
MIN_SAMPLES  = 1


def run_dbscan_clustering():
    """
    Fetches all face embeddings from the DB, runs DBSCAN clustering,
    then reassigns every Face to the correct Person.

    Returns a summary dict for logging/display.
    """
    print("\n========== DBSCAN RESCAN START ==========")

    # --- 1. Load all embeddings ---
    embeddings_qs = Embedding.objects.select_related('face__person').all()

    if not embeddings_qs.exists():
        print("[clusterer] No embeddings found. Aborting.")
        return {"status": "no_data"}

    ids      = []   # embedding IDs (to map back after clustering)
    face_ids = []   # corresponding face IDs
    vectors  = []   # the actual 512-D vectors

    for emb in embeddings_qs:
        if emb.vector is None:
            continue
        ids.append(emb.id)
        face_ids.append(emb.face_id)
        vectors.append(emb.vector)

    print(f"[clusterer] Loaded {len(vectors)} embeddings")

    if len(vectors) < 2:
        print("[clusterer] Need at least 2 embeddings to cluster.")
        return {"status": "too_few"}

    # --- 2. Normalize vectors (converts dot product → cosine similarity) ---
    X = normalize(np.array(vectors, dtype=np.float32))

    # --- 3. Run DBSCAN ---
    # metric='cosine' + normalized vectors gives clean cosine distance
    db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric='cosine', n_jobs=-1)
    labels = db.fit_predict(X)

    unique_labels   = set(labels)
    n_clusters      = len([l for l in unique_labels if l != -1])
    n_noise         = list(labels).count(-1)

    print(f"[clusterer] Clusters found : {n_clusters}")
    print(f"[clusterer] Noise points   : {n_noise}")

    # --- 4. Wipe existing Person assignments ---
    # We're doing a full rescan so we start clean
    Person.objects.all().delete()   # cascades: face.person → NULL via SET_NULL
    print("[clusterer] Cleared all existing Person records")

    # --- 5. Create new Persons from clusters ---
    label_to_person = {}   # cluster label → Person object

    for label in unique_labels:
        if label == -1:
            # Noise: faces that didn't fit any cluster
            # Give each a solo Person so no face is orphaned
            continue
        new_person = Person.objects.create(label=f"Person {label + 1}")
        label_to_person[label] = new_person

    # --- 6. Assign faces + compute avg embeddings ---
    cluster_vectors = {}   # label → list of vectors (for avg calc)

    for idx, label in enumerate(labels):
        face_id = face_ids[idx]
        vector  = vectors[idx]

        if label == -1:
            # Solo person for noise face
            solo_person = Person.objects.create(label="Unknown")
            Face.objects.filter(id=face_id).update(person=solo_person)
            update_person_avg(solo_person, [vector])
        else:
            person = label_to_person[label]
            Face.objects.filter(id=face_id).update(person=person)

            if label not in cluster_vectors:
                cluster_vectors[label] = []
            cluster_vectors[label].append(vector)

    # --- 7. Update avg_embedding for each cluster Person ---
    for label, person in label_to_person.items():
        vecs = cluster_vectors.get(label, [])
        if vecs:
            update_person_avg(person, vecs)

    total_persons = Person.objects.count()
    print(f"[clusterer] Total persons created : {total_persons}")
    print("========== DBSCAN RESCAN END ==========\n")

    return {
        "status"        : "ok",
        "clusters"      : n_clusters,
        "noise"         : n_noise,
        "total_persons" : total_persons,
    }


def update_person_avg(person, vectors):
    """
    Computes and saves the mean embedding from a list of vectors.
    """
    arr = np.array(vectors, dtype=np.float32)
    avg = arr.mean(axis=0).tolist()

    person.avg_embedding = avg
    person.face_count    = len(vectors)
    person.save()