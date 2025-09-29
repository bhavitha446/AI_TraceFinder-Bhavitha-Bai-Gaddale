# app.py
import streamlit as st
import torch, torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import fitz  # PyMuPDF for PDF
import numpy as np
import joblib
import xgboost as xgb
from skimage.feature import local_binary_pattern

# ------------------- Paths -------------------
DOWNLOADS = "C:/Users/bhavi/Downloads"
CNN_CHECKPOINT = f"{DOWNLOADS}/HybridCNN_embed.pth"
XGB_MODEL = f"{DOWNLOADS}/xgb_hybrid_model.json"
SCALER = f"{DOWNLOADS}/hybrid_scaler.pkl"
FORGERY_MODEL = f"{DOWNLOADS}/resnet18_forgery.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- Load models -------------------
# CNN feature extractor
chk = torch.load(CNN_CHECKPOINT, map_location=DEVICE)
classes = chk["classes"]
cnn_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
cnn_model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
cnn_model.fc = nn.Identity()
cnn_model.load_state_dict(chk["state_dict"], strict=False)
cnn_model = cnn_model.to(DEVICE)
cnn_model.eval()

# XGBoost model + scaler
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_MODEL)
scaler = joblib.load(SCALER)

# Forgery model
forgery_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
forgery_model.fc = nn.Linear(forgery_model.fc.in_features, 2)
forgery_model.load_state_dict(torch.load(FORGERY_MODEL, map_location=DEVICE))
forgery_model = forgery_model.to(DEVICE)
forgery_model.eval()
forgery_classes = ["Original", "Tampered"]

# ------------------- Transforms -------------------
cnn_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])
forgery_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

# ------------------- Helper functions -------------------
def pdf_to_images(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    imgs = []
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        imgs.append(img.convert("L"))
    return imgs

def extract_hybrid_features(img):
    # CNN embedding
    img_t = cnn_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = cnn_model(img_t).cpu().numpy().flatten()

    # Handcrafted features (FFT + LBP)
    arr = np.array(img.resize((224,224)).convert("L"))
    F = np.fft.fft2(arr)
    Fm = np.abs(np.fft.fftshift(F))
    fft_mean = float(Fm.mean())
    fft_std = float(Fm.std())

    lbp = local_binary_pattern(arr.astype(np.uint8), P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0,11), range=(0,10))
    hist = hist.astype(float) / (hist.sum() + 1e-9)

    # Final feature vector = CNN (512) + handcrafted (12) = 524
    feat = np.concatenate([emb, [fft_mean, fft_std], hist])
    return feat.reshape(1, -1)

def predict_scanner(img):
    feat = extract_hybrid_features(img)
    feat_scaled = scaler.transform(feat)
    pred = xgb_model.predict(feat_scaled)
    pred_proba = xgb_model.predict_proba(feat_scaled).max()
    return classes[pred[0]], float(pred_proba)

def predict_forgery(img):
    img_t = forgery_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = forgery_model(img_t)
        pred = out.argmax(1).item()
        prob = torch.softmax(out,1)[0,pred].item()
    return forgery_classes[pred], float(prob)

# ------------------- Streamlit UI -------------------
st.title("📑 Scanner Identification + Forgery Detection")

uploaded_file = st.file_uploader(
    "Upload PDF / TIFF / JPG / PNG", 
    type=["pdf","tif","tiff","jpg","jpeg","png"]
)

if uploaded_file:
    st.write(f"Uploaded file: {uploaded_file.name}")
    
    # Convert PDF to images if needed
    if uploaded_file.name.lower().endswith(".pdf"):
        images = pdf_to_images(uploaded_file)
    else:
        img = Image.open(uploaded_file)
        images = [img.convert("L")]
    
    # Process all pages/images
    for i, img in enumerate(images):
        scanner, scanner_acc = predict_scanner(img)
        forgery, forgery_acc = predict_forgery(img)

        st.image(img, caption=f"Page {i+1}", use_column_width=True)
        st.success(f"**Scanner Model:** {scanner} ({scanner_acc*100:.2f}%)")
        st.warning(f"**Forgery Detection:** {forgery} ({forgery_acc*100:.2f}%)")



