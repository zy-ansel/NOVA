import os
import json
import torch
import clip
from PIL import Image
from tqdm import tqdm

############################################################
# Config (fill these paths for your environment)
############################################################

META_JSON = "[PATH_TO_META_DATA_JSON]"
GEN_IMAGE_DIR = "[PATH_TO_GENERATED_IMAGES]"
REF_IMAGE_DIR = "[PATH_TO_REFERENCE_IMAGES]"  # optional, kept for consistency
TASK = "food"  # change if needed

############################################################
# Load metadata
############################################################

with open(META_JSON, "r") as f:
    meta_data = json.load(f)

############################################################
# Load CLIP
############################################################

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

def compute_clip_score(image_path, text):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    text = clip.tokenize([text], truncate=True).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (image_features @ text_features.T).item()
    return similarity

############################################################
# Compute average CLIP score
############################################################

generated_dir = GEN_IMAGE_DIR
total_score = 0.0
count = 0

for root, _, files in os.walk(generated_dir):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(root, file)
            image_id = os.path.splitext(file)[0]

            if image_id in meta_data:
                prompt = meta_data[image_id]["prompt"]
                score = compute_clip_score(image_path, prompt)
                total_score += score
                count += 1
            else:
                print(f"No prompt found for image {image_id}")

if count > 0:
    average_clip_score = total_score / count
    print(f"Average CLIP Score: {average_clip_score:.6f}")
else:
    print("No images were processed.")