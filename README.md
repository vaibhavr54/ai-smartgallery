# AI SmartGallery

**An Intelligent Digital Asset Management System**

> Face recognition · Semantic search · People clustering · Smart collage · Image enhancement · Event grouping — all running locally on your machine. No cloud. No data leaves your device.

---

## ✨ Features

### Core AI Pipeline
| Feature | Description | Model |
|---|---|---|
| **Face Detection** | Detects every face with bounding box | InsightFace Buffalo_L (RetinaFace) |
| **Face Recognition** | 512-D embedding extraction per face | ArcFace ResNet-50 |
| **Person Matching** | Cosine similarity matching on upload | Custom matcher |
| **DBSCAN Clustering** | Unsupervised re-clustering of full library | scikit-learn DBSCAN |
| **Semantic Search** | Natural language photo search | OpenAI CLIP ViT-B-32 |
| **Visual Search** | Upload a face → find all matching photos | InsightFace embeddings |

### Application Features
- 📸 **Gallery** — Browse all photos with face count badges and filter tabs
- 👥 **People View** — Grid of all identified persons with face thumbnails, photo counts, inline rename
- 📅 **Events** — EXIF timestamp-based automatic photo grouping into named events
- ✨ **Image Enhancer** — Real-ESRGAN x4plus super-resolution (2× upscale, GPU accelerated)
- 🖼️ **AI Smart Collage** — Select photos, AI decides layout by visual weight, face-aware cropping
- 🔍 **NL Search** — Type plain English, CLIP returns semantically matching photos
- 👁️ **Face Search** — Upload any face photo, find all photos containing that person
- 📤 **Batch Upload** — Upload multiple photos at once, full AI pipeline runs automatically
- 🗑️ **Delete** — Single photo delete or multi-select bulk delete with live stat updates
- 📱 **Responsive UI** — Glassmorphism dark theme, works on desktop and mobile

---

## 🏗️ System Architecture

```
Presentation Layer (Django Templates — Responsive UI)
        │
Application Layer (Django 4.2)
        │
    ┌───┴───────────────────────┐
    │                           │
InsightFace Engine          CLIP + Vision Engine
detector.py                 encoder.py (ViT-B-32)
matcher.py                  collage_engine.py
clusterer.py (DBSCAN)       enhancer.py (Real-ESRGAN)
event_grouping/grouper.py
        │
Data Layer (SQLite)
Photo · Face · Embedding · Person · Event
```

---

## 🗄️ Database Models

```python
Photo       # image path, upload time, CLIP embedding (512-D)
Face        # bbox (x, y, w, h), confidence, FK→Photo, FK→Person
Embedding   # 512-D InsightFace vector, OneToOne→Face
Person      # label, avg_embedding (512-D), face_count
Event       # name, start_time, end_time
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9+
- NVIDIA GPU with CUDA 12.1 (recommended) or CPU
- 4GB+ RAM

### 1. Clone the repo
```
git clone https://github.com/vaibhavr54/ai-smartgallery.git
cd ai-smartgallery
```

### 2. Create virtual environment
```
python -m venv venvv
# Windows
venvv\Scripts\activate
# Linux/Mac
source venvv/bin/activate
```

### 3. Install dependencies
```
cd backend
pip install -r requirements.txt
```

### 4. Download AI models

**InsightFace** (auto-downloads on first run to `~/.insightface/models/buffalo_l/`)

**CLIP ViT-B-32** (download manually):
```
curl.exe -C - -L -o "C:\Users\<username>\ViT-B-32.safetensors" "https://huggingface.co/timm/vit_base_patch32_clip_224.openai/resolve/main/open_clip_model.safetensors"
```
Update path in `backend/ai_engine/clip_search/encoder.py`.

**Real-ESRGAN** (download manually):
```
curl.exe -C - -L -o "C:\Users\<username>\RealESRGAN_x4plus.pth" "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
```
Update path in `backend/ai_engine/enhancer/enhancer.py`.

### 5. Fix basicsr compatibility (PyTorch 2.5+)
Open `venvv/lib/site-packages/basicsr/data/degradations.py` line 8, change:
```
# FROM
from torchvision.transforms.functional_tensor import rgb_to_grayscale
# TO
from torchvision.transforms.functional import rgb_to_grayscale
```

### 6. Run migrations
```
python manage.py migrate
```

### 7. Start the server
```
python manage.py runserver
```

Visit `http://127.0.0.1:8000`

---

## 📁 Project Structure

```
ai-smartgallery/
└── backend/
    ├── ai_engine/
    │   ├── clip_search/
    │   │   └── encoder.py          # CLIP image & text encoding
    │   ├── collage/
    │   │   └── collage_engine.py   # AI layout + face-aware crop
    │   ├── enhancer/
    │   │   └── enhancer.py         # Real-ESRGAN super-resolution
    │   ├── event_grouping/
    │   │   └── grouper.py          # EXIF timestamp clustering
    │   └── face_recognition/
    │       ├── detector.py         # InsightFace face detection
    │       ├── matcher.py          # Cosine similarity matching
    │       └── clusterer.py        # DBSCAN clustering
    ├── gallery/
    │   ├── templates/gallery/      # All HTML templates
    │   ├── models.py               # Photo, Face, Embedding, Person, Event
    │   ├── views.py                # All views + AI pipeline
    │   └── urls.py                 # URL routing
    ├── config/
    │   ├── settings.py
    │   └── urls.py
    └── manage.py
```

---

## 🔬 Research Foundation

| Paper | Authors | Used For |
|---|---|---|
| FaceNet | Schroff et al., Google (2015) | Embedding-based similarity concept |
| ArcFace / InsightFace | Deng et al. (2019) | Face recognition backbone |
| DBSCAN | Ester et al. (1996) | Unsupervised person clustering |
| CLIP | Radford et al., OpenAI (2021) | Natural language image search |
| RetinaFace | Deng et al. (2020) | Face detection + alignment |
| Real-ESRGAN | Wang et al. (2021) | Image super-resolution |

---

## ⚠️ Known Limitations

- Synchronous processing — uploads block HTTP response (fix: Celery + Redis)
- Linear search O(N) — slow at large scale (fix: FAISS indexing)
- English queries only for CLIP NL search
- Single user, no authentication
- Face angle limit ~60° from frontal view

---

## 🔜 Planned Features

- Video Analysis — face recognition in videos
- FAISS indexed search
- Mobile app (TensorFlow Lite)
- Multilingual CLIP

---

## 🛠️ Tech Stack

`Django 4.2` · `InsightFace` · `OpenAI CLIP` · `Real-ESRGAN` · `DBSCAN` · `ONNX Runtime` · `PyTorch 2.5` · `SQLite` · `Python 3.9`

---

*Built entirely on open-source technology. No cloud APIs used at runtime.*