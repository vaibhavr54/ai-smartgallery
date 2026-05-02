import os
import cv2
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

MODEL_PATH = 'C:/Users/vaibh/RealESRGAN_x4plus.pth'
ENHANCED_DIR = 'media/enhanced/'

_upsampler = None

def get_upsampler():
    global _upsampler
    if _upsampler is None:
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23, num_grow_ch=32,
            scale=4
        )
        _upsampler = RealESRGANer(
            scale=4,
            model_path=MODEL_PATH,
            model=model,
            tile=0,        # no tiling on GPU
            tile_pad=10,
            pre_pad=0,
            half=True,     # half precision — faster on GPU
            device='cuda',
        )
        print("[enhancer] Real-ESRGAN loaded on CUDA")
    return _upsampler


def enhance_image(input_path, output_filename):
    os.makedirs(ENHANCED_DIR, exist_ok=True)

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {input_path}")

    upsampler = get_upsampler()
    output, _ = upsampler.enhance(img, outscale=2)

    output_path = os.path.join(ENHANCED_DIR, output_filename)
    cv2.imwrite(output_path, output, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"[enhancer] Enhanced → {output_path}")
    return output_path