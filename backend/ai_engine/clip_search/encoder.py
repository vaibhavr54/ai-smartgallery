import open_clip
import torch
from PIL import Image
from safetensors.torch import load_file

# Load model architecture (no pretrained weights)
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained=None
)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Load weights from local safetensors file
state_dict = load_file('C:/Users/vaibh/ViT-B-32.safetensors')
model.load_state_dict(state_dict, strict=False)
model.eval()

def encode_image(image_path):
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(img)
        embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding[0].tolist()

def encode_text(query):
    tokens = tokenizer([query])
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding[0].tolist()