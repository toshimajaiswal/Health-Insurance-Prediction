# 🏥 Health Insurance Cost Prediction

This project is an end-to-end **Machine Learning web application** that predicts **health insurance charges** based on user details such as age, BMI, smoking habits, and region. The trained model is deployed as an **interactive Streamlit web app**.

---

## 🚀 Live Demo
👉 https://health-insurance-prediction-toshima2612.streamlit.app

---

## 📌 Problem Statement
Health insurance costs depend on multiple personal and lifestyle factors. The goal of this project is to build a machine learning model that can accurately estimate insurance charges and make it accessible through a simple and interactive web interface.

---

## 🧠 Machine Learning Approach
- Data preprocessing and feature engineering  
- Encoding categorical variables  
- Training and comparing multiple regression models  
- Selecting the best-performing model based on evaluation metrics  
- Deploying the final model for real-time predictions  

---

## 📊 Exploratory Data Analysis (EDA)
Exploratory analysis was performed to understand feature distributions and relationships.
Key insights include higher insurance costs for smokers and a strong correlation between BMI and charges.
Detailed EDA and visualizations are available in the Jupyter notebook.

---

## ⚙️ Tech Stack

**Programming Language**
- Python

**Libraries & Tools**
- Pandas  
- NumPy  
- Scikit-learn  
- Joblib  
- Streamlit  

**Deployment**
- Streamlit Cloud  
- GitHub  

---


---

## 🧪 Models Used
- Linear Regression  
- Random Forest Regressor  

The model with the **better R² score** is automatically selected and saved for deployment.

---

## 📊 Evaluation Metrics
- Root Mean Squared Error (RMSE)  
- R² Score  

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/health-insurance-prediction.git
cd health-insurance-prediction
```

### 2️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 3️⃣ Train the model
```
python train.py
```

### 4️⃣ Run the Streamlit app
```
streamlit run app.py
```

## 🌐 Web Application Features
- User-friendly input form
- Real-time insurance cost prediction
- Consistent preprocessing between training and inference
- Deployed and accessible online

## 🙌 Acknowledgements
This project was built to understand the complete machine learning lifecycle, from model development to real-world deployment.

