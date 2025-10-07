🧠 AI TraceFinder: Scanner Identification & Forgery Detection
🔍 Overview

AI TraceFinder is an advanced AI-powered system designed to identify the source scanner of a document and detect image forgeries such as copy-move, splicing, or retouching.
It combines deep learning (ResNet18) and handcrafted statistical features (FFT, LBP) in a hybrid model, achieving high accuracy in both scanner fingerprinting and tampering classification.

⚙️ Features

✅ Scanner Model Identification — Detects which scanner produced a document based on PRNU and CNN embeddings
✅ Forgery Detection — Classifies document images as Original or Tampered using a fine-tuned ResNet18
✅ Hybrid Feature Extraction — Combines deep CNN features with FFT and LBP handcrafted statistics
✅ Supports Multiple Formats — Works with PDF, TIFF, JPEG, and PNG files
✅ Streamlit Interface — Simple drag-and-drop web app for inference
✅ Automated Model Downloads — All models auto-download from GitHub if not present locally

🧩 System Architecture
📂 AI_TraceFinder
│
├── application.py                     # Streamlit frontend (main application)
├── HybridCNN_embed.pth        # Trained CNN model (feature extractor)
├── xgb_hybrid_model.json      # XGBoost model for scanner classification
├── hybrid_scaler.pkl          # Feature scaler for XGBoost inputs
├── resnet18_forgery.pth       # ResNet18 forgery detection model
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

🔧 Workflow

Input Upload → PDF or image file

PDF-to-Image Conversion using PyMuPDF (fitz)

Hybrid Feature Extraction

CNN Embedding (ResNet18 backbone)

FFT Statistical Metrics (mean, std)

LBP Histogram (texture descriptor)

Scanner Identification via XGBoost Classifier

Forgery Detection using fine-tuned ResNet18

Results Visualization in Streamlit

🧠 Models Used
Model Type	Architecture	Purpose
HybridCNN_embed.pth	Modified ResNet18	CNN embedding extractor
xgb_hybrid_model.json	XGBoost Classifier	Scanner identification
hybrid_scaler.pkl	StandardScaler	Normalization of hybrid features
resnet18_forgery.pth	Fine-tuned ResNet18	Forgery (tampering) detection
💻 Installation & Usage
1️⃣ Clone the Repository
git clone https://github.com/bhavitha446/AI_TraceFinder.git
cd AI_TraceFinder

2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the App
streamlit run app.py

5️⃣ Upload Input

Upload a PDF, TIFF, JPEG, or PNG file.

The app automatically:

Converts PDF pages to images

Extracts hybrid features

Predicts scanner model

Detects forgery status

Displays probability and confidence levels

🧮 Technologies Used
Category	Libraries/Tools
Frontend	Streamlit
Deep Learning	PyTorch, torchvision
Machine Learning	XGBoost, scikit-learn
Image Processing	NumPy, scikit-image, PIL, PyMuPDF
Utilities	joblib, requests, os

🧠 Example Output

Input: Scanned PDF or TIFF document
Output:

📄 Page 1
✅ Scanner Model: Canon_LiDE300 (98.54%)
⚠️ Forgery Detection: Tampered (91.72%)


Displays image preview + prediction scores for each page.

📊 Experimental Summary

Dataset: Flatfield, Official, and Wikipedia scans (150 & 300 dpi)

Features: PRNU, FFT (mean/std), LBP Histogram, CNN Embeddings

Models: Logistic Regression, SVM, Random Forest, XGBoost, ResNet18

Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

🚀 Future Scope

Integration of CNN + Transformer hybrids (ViT)

Forgery localization maps using segmentation networks

Web API service for document authenticity verification

Mobile app for field authentication

👩‍💻 Author

Bhavitha Bai Gaddale
AI Researcher | Computer Vision & Document Forensics
bhavithagaddale21@gmail.com
