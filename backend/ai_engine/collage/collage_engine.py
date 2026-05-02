import os
import math
import numpy as np
from PIL import Image, ImageDraw

COLLAGE_DIR = 'media/collages/'
CELL_SIZE   = 400   # each grid cell is 400x400px
GAP         = 8     # gap between photos in pixels
BG_COLOR    = (10, 12, 22)  # dark background matching UI


def compute_visual_weight(photo, faces):
    """
    Score a photo by how visually important it is.
    More faces + larger faces = higher weight = bigger cell.
    """
    if not faces:
        return 1.0

    face_area_sum = sum(f.width * f.height for f in faces)

    try:
        img = Image.open(photo.image.path)
        photo_area = img.width * img.height
    except Exception:
        photo_area = 500 * 500

    ratio = face_area_sum / max(photo_area, 1)
    weight = 1.0 + (len(faces) * 0.4) + (ratio * 3.0)
    return min(weight, 3.5)


def decide_layout(n_photos, weights):
    """
    Returns list of (row_span, col_span) for each photo.
    Total grid columns = 3 for most layouts.
    """
    COLS = 3

    # Sort indices by weight descending
    order = sorted(range(n_photos), key=lambda i: weights[i], reverse=True)

    spans = [None] * n_photos

    # Top photo gets featured spot if weight is significantly higher
    if n_photos == 1:
        spans[0] = (2, 3)
    elif n_photos == 2:
        spans[order[0]] = (2, 2)
        spans[order[1]] = (2, 1)
    elif n_photos == 3:
        spans[order[0]] = (2, 2)
        spans[order[1]] = (1, 1)
        spans[order[2]] = (1, 1)
    elif n_photos == 4:
        spans[order[0]] = (2, 2)
        for i in order[1:]:
            spans[i] = (1, 1)
    elif n_photos <= 6:
        spans[order[0]] = (2, 2)
        for i in order[1:]:
            spans[i] = (1, 1)
    else:
        # All equal for large counts
        for i in range(n_photos):
            spans[i] = (1, 1)

    return spans, COLS


def smart_crop(img, target_w, target_h, faces):
    """
    Crop image to target_w x target_h.
    If faces exist, center crop around face centroid.
    Otherwise use center crop.
    """
    src_w, src_h = img.size

    # Compute scale to fill target
    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    if faces:
        # Face centroid in original image coords
        cx = sum(f.x + f.width  // 2 for f in faces) / len(faces)
        cy = sum(f.y + f.height // 2 for f in faces) / len(faces)

        # Scale centroid to resized image
        cx = int(cx * scale)
        cy = int(cy * scale)

        # Compute crop box centered on faces
        left = max(0, min(cx - target_w // 2, new_w - target_w))
        top  = max(0, min(cy - target_h // 2, new_h - target_h))
    else:
        # Center crop
        left = (new_w - target_w) // 2
        top  = (new_h - target_h) // 2

    return img.crop((left, top, left + target_w, top + target_h))


def build_collage(photos_with_faces, output_filename):
    """
    Main entry point.
    photos_with_faces: list of (Photo, [Face, ...])
    Returns output path.
    """
    os.makedirs(COLLAGE_DIR, exist_ok=True)

    n = len(photos_with_faces)
    if n == 0:
        raise ValueError("No photos selected")

    weights = [compute_visual_weight(p, f) for p, f in photos_with_faces]
    spans, COLS = decide_layout(n, weights)

    # ── Build grid using a simple packing algorithm ──
    # Track occupied cells
    ROWS = math.ceil(sum(rs * cs for rs, cs in spans) / COLS) + 2
    grid = [[None] * COLS for _ in range(ROWS)]

    placed = []  # (photo_idx, row, col, row_span, col_span)

    def can_place(r, c, rs, cs):
        if c + cs > COLS: return False
        if r + rs > ROWS: return False
        for dr in range(rs):
            for dc in range(cs):
                if grid[r + dr][c + dc] is not None:
                    return False
        return True

    def place(idx, r, c, rs, cs):
        for dr in range(rs):
            for dc in range(cs):
                grid[r + dr][c + dc] = idx
        placed.append((idx, r, c, rs, cs))

    # Place photos in weight order
    order = sorted(range(n), key=lambda i: weights[i], reverse=True)

    for idx in order:
        rs, cs = spans[idx]
        placed_flag = False
        for r in range(ROWS):
            for c in range(COLS):
                if can_place(r, c, rs, cs):
                    place(idx, r, c, rs, cs)
                    placed_flag = True
                    break
            if placed_flag:
                break
        if not placed_flag:
            # Fallback: place as 1x1
            for r in range(ROWS):
                for c in range(COLS):
                    if can_place(r, c, 1, 1):
                        place(idx, r, c, 1, 1)
                        break
                else:
                    continue
                break

    # Find actual used rows
    used_rows = max(r + rs for _, r, _, rs, _ in placed)

    # ── Compose canvas ──
    canvas_w = COLS * CELL_SIZE + (COLS + 1) * GAP
    canvas_h = used_rows * CELL_SIZE + (used_rows + 1) * GAP

    canvas = Image.new('RGB', (canvas_w, canvas_h), BG_COLOR)

    for idx, row, col, rs, cs in placed:
        photo, faces = photos_with_faces[idx]

        cell_w = cs * CELL_SIZE + (cs - 1) * GAP
        cell_h = rs * CELL_SIZE + (rs - 1) * GAP

        try:
            img = Image.open(photo.image.path).convert('RGB')
            cropped = smart_crop(img, cell_w, cell_h, faces)
        except Exception as e:
            print(f"[collage] Error processing photo {photo.id}: {e}")
            cropped = Image.new('RGB', (cell_w, cell_h), (30, 35, 60))

        x = GAP + col * (CELL_SIZE + GAP)
        y = GAP + row * (CELL_SIZE + GAP)
        canvas.paste(cropped, (x, y))

    output_path = os.path.join(COLLAGE_DIR, output_filename)
    canvas.save(output_path, 'JPEG', quality=93)
    print(f"[collage] Saved → {output_path}")
    return output_path