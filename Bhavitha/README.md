🧠 AI TraceFinder: Scanner Identification & Forgery Detection 🔍
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
├── application.py              # Streamlit frontend (main application)
├── HybridCNN_embed.pth         # Trained CNN model (feature extractor)
├── xgb_hybrid_model.json       # XGBoost model for scanner classification
├── hybrid_scaler.pkl           # Feature scaler for XGBoost inputs
├── resnet18_forgery.pth        # ResNet18 forgery detection model
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

🔧 Workflow

1️⃣ Input Upload → PDF or image file
2️⃣ PDF-to-Image Conversion using PyMuPDF (fitz)
3️⃣ Hybrid Feature Extraction

CNN Embedding (ResNet18 backbone)

FFT Statistical Metrics (mean, std)

LBP Histogram (texture descriptor)
4️⃣ Scanner Identification via XGBoost Classifier
5️⃣ Forgery Detection using fine-tuned ResNet18
6️⃣ Results Visualization in Streamlit

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
source venv/bin/activate       # On macOS/Linux
venv\Scripts\activate          # On Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the App
streamlit run application.py

5️⃣ Upload Input

Upload a PDF, TIFF, JPEG, or PNG file. The app automatically:

Converts PDF pages to images

Extracts hybrid features

Predicts scanner model

Detects forgery status

Displays probability and confidence levels

🧮 Technologies Used
Category	Libraries / Tools
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


The app displays the image preview and prediction scores for each page.

📊 Experimental Summary

Dataset: Flatfield, Official, and Wikipedia scans (150 & 300 dpi)

Features: PRNU, FFT (mean/std), LBP Histogram, CNN Embeddings

Models: Logistic Regression, SVM, Random Forest, XGBoost, ResNet18

Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

📘 Project Documentation
i) Project Objective / Overview

The objective of AI TraceFinder is to identify the source scanner device used to scan an image and detect document tampering.
Each scanner introduces unique noise and frequency patterns that can be analyzed through deep learning and handcrafted feature extraction.
This project supports digital forensics, document authentication, and legal verification.

ii) System Design and Architecture

The system consists of five core stages:

Data Collection & Labeling – Collect scanned samples from multiple scanner models.

Image Preprocessing – Resize, denoise, normalize, and convert to grayscale.

Feature Extraction – Extract FFT, LBP, PRNU, and CNN embeddings.

Model Training & Evaluation – Train SVM, Random Forest, and CNN models; evaluate with accuracy, F1-score, and confusion matrix.

Deployment System – Streamlit web app to upload and predict scanner model with confidence score.

iii) Data Collection, Management, and Processing

Collection: Gathered scanned images from different scanners.

Management: Labeled and structured dataset into folders per scanner type.

Processing: Resized, denoised, normalized, and extracted noise features using FFT/Wavelet filters.
This ensures consistent and high-quality training data.

iv) Technical Approach (AI / ML / etc.)

Uses a hybrid AI approach:

Feature-Based ML: FFT, LBP, and PRNU features fed into XGBoost and SVM models.

Deep Learning: ResNet18 CNN trained for direct scanner identification and forgery detection.

Explainable AI: Grad-CAM and SHAP for visual interpretation of decisions.

v) Innovation or Problem-Solving Aspect

Introduces scanner source identification as a digital forensic solution.

Combines handcrafted and deep learning features for improved accuracy.

Uses explainable AI tools (Grad-CAM/SHAP) for trust and transparency.

Streamlit-based real-time web app for non-technical users.

vi) Team Collaboration and Project Management

Followed Agile methodology with 8-week milestones.

Roles:

Data Engineer – Dataset collection and labeling

ML Developer – Model training and feature engineering

System Architect – Workflow design

Frontend Lead – Streamlit deployment

Tools: GitHub, Trello, Google Drive for collaboration and tracking.

vii) Project Impact and Future Scope

Impact:

Supports forensic document verification and prevents digital forgery.

Enhances authenticity checks in legal, educational, and corporate sectors.

Future Scope:

Extend model for camera and printer identification.

Cloud and mobile deployment for scalable verification.

Integrate blockchain for tamper-proof audit logs.

viii) Key Learnings

Gained expertise in FFT, LBP, PRNU, and CNN architectures.

Understood the role of preprocessing and data balance in model success.

Learned team coordination, Agile workflow, and deployment with Streamlit.

Improved ability to interpret models using Grad-CAM/SHAP.

Strengthened skills in AI for digital forensics and document analysis.

🚀 Future Enhancements

Integrate CNN + Transformer (ViT) hybrid networks.

Implement forgery localization maps using segmentation models.

Develop RESTful API and Android app for on-site verification.

👩‍💻 Author

Bhavitha Bai Gaddale
AI Researcher | Computer Vision & Document Forensics
📧 bhavithagaddale21@gmail.com
