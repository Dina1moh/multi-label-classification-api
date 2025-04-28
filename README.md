# 🖼️ Multi-Label Image Classification API

This project builds a **Multi-Label Image Classifier** for detecting multiple objects in a single image based on the **PASCAL VOC 2007** dataset. It uses **Transfer Learning** with **ResNet50** and provides a **FastAPI** interface for prediction.

---

## 🚀 Project Structure
```
src/
├── artifacts/           # Trained model (best_model.pth)
├── models/              # Model architecture (model.py)
├── utils/ src              # Utility functions (predict.py, transforms.py)
├── main.py
|__inference.py              # FastAPI app entry point
├── requirements.txt     # Python dependencies
├── README.md            # Project description
```

---

##  How It Works

1. **Training**:  
   - Used **ResNet50** pretrained on ImageNet.
   - Modified the final layer to predict multiple labels.
   - Trained on **PASCAL VOC 2007** dataset.

2. **Serving**:  
   - The trained model is loaded into a **FastAPI** server.
   - You can upload an image through a simple HTML form.
   - The API returns the top predicted labels with confidence scores.

---

## 🛠️ Setup

1. Clone the repository:
   ```
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the API:
   ```
   uvicorn src.main:app --reload
   ```

5. Open your browser at:
   ```
   http://127.0.0.1:8000
   ```

---

## 🧠 Model Details

- **Base Model**: ResNet50
- **Dataset**: PASCAL VOC 2007
- **Task**: Multi-Label Classification
- **Loss Function**: Binary Cross Entropy (BCEWithLogitsLoss)
- **Optimizer**: Adam

---

## 📸 How to Use

- Go to `http://127.0.0.1:8000`
- Upload an image.
- Get the predicted labels and their probabilities.

---

## 📂 Example Prediction

| Label         | Probability |
|---------------|-------------|
| Dog           | 97%         |
| Car           | 85%         |
| Person        | 90%         |

---

##  Acknowledgements

- [PASCAL VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)
- [PyTorch](https://pytorch.org/)
- [FastAPI](https://fastapi.tiangolo.com/)

---
