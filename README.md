# 🚢 Titanic Survival Prediction System

A professional, deep learning-based web application to predict passenger survival on the Titanic using Artificial Neural Networks (ANN).

## 📁 Project Structure
- `app.py`: Premium Streamlit application with glassmorphism UI.
- `train.py`: Training script to reproduce the ANN model and scaler.
- `model/`: Directory containing the saved TensorFlow model and MinMaxScaler.
- `data/`: Directory for the Titanic dataset.
- `requirements.txt`: Python dependencies.

## 🚀 How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Train the Model
If you need to regenerate the model files:
```bash
python train.py
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

## 🌐 Deployment (Streamlit Community Cloud)
1. Push this repository to GitHub.
2. Sign in to [Streamlit Cloud](https://share.streamlit.io/).
3. Connect your GitHub repo.
4. Select `app.py` as the main file and deploy!

## 🛠️ Tech Stack
- **Backend:** TensorFlow (ANN), Scikit-learn (Preprocessing)
- **Frontend:** Streamlit, Custom CSS (Glassmorphism)
- **Visuals:** Plotly (Interative Charts)
- **Language:** Python
