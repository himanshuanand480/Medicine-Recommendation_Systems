# 💊 AI Health Advisor — End-to-End Machine Learning Project

This project predicts diseases based on user-input symptoms and recommends medicines, diet, precautions, and workout plans using machine learning.

---

## 🧠 Problem Statement
Access to reliable preliminary medical guidance is limited. This system helps users identify possible diseases and provides basic health recommendations using AI-driven prediction.

---

## ⚙️ Tech Stack
- **Language:** Python  
- **Libraries:** scikit-learn, Pandas, NumPy, Seaborn, Streamlit  
- **Model:** Random Forest Classifier  
- **Tools:** VS Code / PyCharm, GitHub, Streamlit  

---

## 🚀 Features
- Takes multiple symptom inputs and predicts the most probable disease.  
- Displays relevant **medicines, diet, precautions, and workouts**.  
- Interactive **Streamlit** web app with real-time responses.  
- Clean UI and efficient caching for faster performance.

---

## 📊 Dataset & Preprocessing
- Source: Custom CSV dataset mapping symptoms → diseases.  
- Handled missing data, standardized feature names.  
- Label encoded categorical values for ML models.  

---

## 🧩 Model Training
- **Algorithm:** Random Forest Classifier  
- **Accuracy:** ~92–94%  
- Trained on symptom-disease pairs; validated with test set.

---

## 🖥️ Demo
Clone and run locally:
```bash
git clone https://github.com/himanshuanand480/Medicine-Recommendation_Systems.git
cd Medicine-Recommendation_Systems
pip install -r requirements.txt
streamlit run app.py
