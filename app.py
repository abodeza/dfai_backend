# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel
import os

import clip
import torch
from PIL import Image

app = FastAPI()

def _norm(x): 
    return x / x.norm(dim=-1, keepdim=True)

def _sim(a, b):
    # a: (1, D) text, b: (1, D) image
    return float((a.cpu().numpy() @ b.cpu().numpy().T).squeeze())

def evaluate_one(name, img_feat):
    img_n = _norm(img_feat)

    s_ulcer      = _sim(_norm(ulcer),      img_n)
    s_no_ulcer   = _sim(_norm(no_ulcer),   img_n)
    s_abscess    = _sim(_norm(abscess),    img_n)
    s_no_abscess = _sim(_norm(no_abscess), img_n)
    s_gangrene   = _sim(_norm(gangrene),   img_n)
    s_no_gangrene= _sim(_norm(no_gangrene),img_n)

    ulcer_exists    = int(s_ulcer > s_no_ulcer)
    abscess_exists  = int(s_abscess > s_no_abscess)
    gangrene_exists = int(s_gangrene > s_no_gangrene)

    # urgency precedence
    if gangrene_exists == 1:
        urgency = "critical"
    elif ulcer_exists == 1 and abscess_exists == 1:
        urgency = "high"
    elif (ulcer_exists ^ abscess_exists) == 1:
        urgency = "moderate"
    else:
        urgency = "low"

    return [name, ulcer_exists, abscess_exists, gangrene_exists, urgency]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500"],   # include 'null' for file://, '*' for anything else
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_32, preprocess = clip.load("ViT-B/32", device=device)
model_32.eval()

with torch.no_grad():
    no_ulcer = model_32.encode_text(clip.tokenize("a healthy human foot with intact skin and no wounds").to(device)).float()
    ulcer = model_32.encode_text(torch.tensor(clip.tokenize("a human foot wound").to(device))).float()

    no_abscess = model_32.encode_text(clip.tokenize("a human foot with no abscess on wound").to(device)).float()
    abscess = model_32.encode_text(torch.tensor(clip.tokenize("a human foot with abscess covering wound").to(device))).float()

    no_gangrene = model_32.encode_text(clip.tokenize("a human foot with no dead black tissue").to(device)).float()
    gangrene = model_32.encode_text(torch.tensor(clip.tokenize("a human foot with black dead tissue and gangrene").to(device))).float()

print("checkpoint: text features encoded")

class PredictRequest(BaseModel):
    filename: str

@app.post("/predict")
def predict(req: PredictRequest):
    # check file exists in test_imgs
    img_path = os.path.join("test_imgs", req.filename)
    if not os.path.isfile(img_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {req.filename}")

    # load and preprocess
    input_image = Image.open(img_path).convert("RGB")
    preprocessed_image = preprocess(input_image)
    print("checkpoint: image loaded")

    # encode image
    with torch.no_grad():
        preprocessed_image_feat = model_32.encode_image(torch.tensor(preprocessed_image).to(device).unsqueeze(0)).float()
    print("checkpoint: image feature encoded")

    # evaluate
    p = evaluate_one(req.filename, preprocessed_image_feat)
    print("checkpoint: evaluation done")

    # return list as-is
    return p

if __name__ == "__main__":
    # simple local runner; on AWS you can run with uvicorn as well
    import uvicorn
    uvicorn.run("app:predict", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
