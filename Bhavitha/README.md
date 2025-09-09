# AI TraceFinder – Scanner & Manipulation Detection

This project implements **digital image forensic analysis** using the **Supatlantique Scanned Documents Dataset**.  
We detect the **scanner model** and **forgeries (manipulations)** using a **hybrid pipeline** that combines:

-  **PRNU Fingerprints** (scanner noise pattern extraction)  
-  **FFT Features** (frequency-based statistics)  
-  **Hybrid CNN (ResNet34 embeddings + handcrafted features)**  
-  **XGBoost Fusion Model** (achieved >91% accuracy on scanner identification)  
-  **U-Net Segmentation** (for tampering localization)

---

##  Dataset
We used the **Supatlantique Scanned Documents Database**:
- `Flatfield/` → Reference PRNU fingerprints  
- `Official/` → Genuine scanned images  
- `Wikipedia/` → Public-sourced scanned images  
- `Tampered images/` → For manipulation detection  

---

##  Pipeline / Milestones
1. **Mount Drive & Explore Dataset**  
2. **PRNU Extraction** (from flatfield images using wavelet denoising)  
3. **FFT + PRNU Feature Extraction**  
4. **Classical ML Models** (SVM, RF, LR → baseline ~68%)  
5. **Hybrid CNN** (ResNet34 + handcrafted features → ~79%)  
6. **XGBoost Fusion** (CNN + FFT features → ~91%) ✅  
7. **Forgery Detection**  
   - Classification: ResNet34  
   - Localization: U-Net segmentation  
8. **Deployment**: Gradio App with  
   - Single image upload → predicted scanner / forgery  
   - Batch upload → CSV report + accuracy  

---

##  Requirements
- Python 3.10+  
- PyTorch / Torchvision  
- OpenCV  
- Scikit-learn  
- XGBoost  
- Gradio  

Or run directly on **Google Colab (GPU)**.

---

##  Training
```bash
# Extract PRNU + FFT features
python extract_features.py

# Train hybrid CNN
python train_hybrid_cnn.py

# Train XGBoost fusion
python train_xgboost.py

# Train U-Net for tampering localization
python train_unet.py
