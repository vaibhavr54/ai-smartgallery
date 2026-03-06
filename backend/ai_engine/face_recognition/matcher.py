# ai_engine/face_recognition/matcher.py

import numpy as np
from numpy.linalg import norm

# How similar two embeddings must be to be considered the same person.
# 0.50 is a solid starting point — lower = stricter, higher = more lenient
SIMILARITY_THRESHOLD = 0.35


def cosine_similarity(vec_a, vec_b):
    """
    Returns a float between -1 and 1.
    1.0  = identical faces
    0.0  = completely unrelated
    """
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)

    denom = norm(a) * norm(b)
    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


def find_matching_person(new_embedding, all_persons):
    """
    Compares a new face embedding against every known Person's average embedding.
    Returns the best matching Person object, or None if no match is found.

    Parameters:
        new_embedding  : list of 512 floats (from detector)
        all_persons    : queryset of Person objects that have avg_embedding set

    Returns:
        Person object or None
    """
    best_person = None
    best_score  = -1.0

    for person in all_persons:
        if not person.avg_embedding:
            continue

        score = cosine_similarity(new_embedding, person.avg_embedding)

        if score > best_score:
            best_score  = score
            best_person = person

    if best_score >= SIMILARITY_THRESHOLD:
        print(f"[matcher] Matched Person {best_person.id} with score {best_score:.4f}")
        return best_person

    print(f"[matcher] No match found (best score was {best_score:.4f}) — creating new Person")
    return None


def update_average_embedding(person, new_embedding):
    """
    Updates a Person's average embedding using an incremental mean formula.
    This avoids storing every single embedding and recomputing from scratch.

    Formula:  new_avg = (old_avg * n + new_vec) / (n + 1)

    Parameters:
        person        : Person model instance (must have avg_embedding + face_count)
        new_embedding : list of 512 floats
    """
    new_vec = np.array(new_embedding, dtype=np.float32)

    if person.avg_embedding is None:
        # First face for this person
        person.avg_embedding = new_vec.tolist()
        person.face_count    = 1
    else:
        old_avg = np.array(person.avg_embedding, dtype=np.float32)
        n       = person.face_count or 1

        updated = (old_avg * n + new_vec) / (n + 1)

        person.avg_embedding = updated.tolist()
        person.face_count    = n + 1

    person.save()
    print(f"[matcher] Updated avg embedding for Person {person.id} (face_count={person.face_count})")