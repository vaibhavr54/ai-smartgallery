# ai_engine/face_recognition/detector.py

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ctx_id=0 means GPU. If you don't have CUDA, change to ctx_id=-1 for CPU
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1)

def detect_faces(image_path):
    """
    Runs InsightFace on an image and returns a list of face dicts.
    Each dict contains bbox, embedding (512-D), and detection score.
    """
    img = cv2.imread(image_path)

    if img is None:
        print(f"[detector] Could not read image: {image_path}")
        return []

    faces = app.get(img)

    results = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)

        embedding = None
        if face.embedding is not None:
            embedding = face.embedding.tolist()  # convert numpy → plain list for JSON storage

        results.append({
            "x":          int(x1),
            "y":          int(y1),
            "width":      int(x2 - x1),
            "height":     int(y2 - y1),
            "det_score":  float(face.det_score),
            "embedding":  embedding,       # 512-D list, or None if model failed
        })

    return results