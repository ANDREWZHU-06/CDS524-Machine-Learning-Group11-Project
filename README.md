# Lingnan University, Hong Kong
# CDS524-Machine-Learning-Group11-Project

# **Diabetes Detection System - Machine Learning Project**

## **Introduction**
This repository contains a **Machine Learning-based Diabetes Detection System**, developed as a group project for **CDS524 - Machine Learning for Business** at **Lingnan University**. The project aims to predict diabetes risk based on clinical and demographic data using multiple machine learning models, including **Logistic Regression, Random Forest, Gradient Boosting, SVM, Neural Networks, and XGBoost**.  

The system provides:
- **Diabetes risk prediction** based on clinical data.
- **Feature selection and model comparison** to choose the best-performing algorithm.
- **Integration with DeepSeek AI API** for enhanced treatment recommendations.
- **A Streamlit-based Web Application** for user-friendly interaction.

---

## **Table of Contents**
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [System Architecture](#system-architecture)
- [Machine Learning Models](#machine-learning-models)
- [API Integration](#api-integration)
- [Results and Visualizations](#results-and-visualizations)
- [Contributors](#contributors)
- [License](#license)

---

## **Project Overview**
This system is designed to:
- **Analyze patient data** (age, BMI, blood test results, etc.).
- **Train multiple ML models** to predict diabetes risk.
- **Compare models** based on accuracy, precision, recall, F1-score, and ROC-AUC.
- **Provide AI-enhanced treatment recommendations** using DeepSeek AI.
- **Deploy a web application** for real-time diabetes risk assessment.

---

## **Features**
✅ **Data Preprocessing**
   - Handling missing values using **SimpleImputer & KNNImputer**.
   - Detecting and handling **outliers using IQR method**.
   - Encoding categorical variables (e.g., **Gender, CLASS**).

✅ **Feature Selection**
   - Using **ANOVA F-score** and **Random Forest feature importance**.

✅ **Machine Learning Model Training**
   - **Logistic Regression**
   - **Random Forest**
   - **Gradient Boosting**
   - **SVM**
   - **Neural Networks**
   - **XGBoost**
   - **Hyperparameter tuning with GridSearchCV**.

✅ **Model Evaluation**
   - Performance metrics: **Accuracy, Precision, Recall, F1-score, ROC-AUC**.
   - **Confusion Matrix & Classification Reports**.
   - **ROC Curves & Learning Curves** for visualization.

✅ **AI-Enhanced Treatment Recommendations**
   - Integration with **DeepSeek AI API**.
   - **Fallback system** for offline recommendations.

✅ **Web Application**
   - Built using **Streamlit**.
   - User-friendly interface for **diabetes risk prediction**.

---

## **Dataset**
The dataset used for this project is **Dataset of Diabetes_updated.csv**, which contains:
- **Clinical data**: HbA1c, Cholesterol, Triglycerides, HDL, LDL, VLDL, Urea, Creatinine, BMI.
- **Demographic data**: Age, Gender.
- **Target variable**: CLASS (N = Normal, P = Pre-Diabetic, Y = Diabetic).

---

## **Installation**
### **1️⃣ Clone the repository**
```bash
git clone https://github.com/your-repo/diabetes-detection.git
cd diabetes-detection
```

### **2️⃣ Set up a virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### **3️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Run the application**
```bash
streamlit run app_v2.py
```

---

## **Usage**
### **1️⃣ Running the Machine Learning Pipeline**
```bash
python main.py
```
This script will:
- Load and preprocess the dataset.
- Train multiple machine learning models.
- Perform feature selection and model evaluation.
- Save the best-performing model.

### **2️⃣ Running the Streamlit Web App**
```bash
streamlit run app_v2.py
```
This will launch a **web application** for real-time diabetes risk assessment.

---

## **System Architecture**
```
📂 Diabetes-Detection-Project
│── 📂 models/               # Trained Machine Learning models (saved as .pkl)
│── 📂 results/              # Visualization and model performance reports
│── 📂 data/                 # Dataset files (CSV)
│── 📂 notebooks/            # Jupyter notebooks for EDA and model training
│── 📜 main.py               # Core ML pipeline (data processing, training, evaluation)
│── 📜 api_v1.py             # API integration for treatment recommendations
│── 📜 app_v2.py             # Streamlit Web App for Diabetes Risk Assessment
│── 📜 README.md             # Documentation
│── 📜 requirements.txt      # Python dependencies
```

---

## **Machine Learning Models**
### **1️⃣ Logistic Regression**
- Simple, interpretable model.
- Suitable for linear relationships.

### **2️⃣ Random Forest**
- Handles non-linear relationships well.
- Provides feature importance.

### **3️⃣ Gradient Boosting (XGBoost)**
- High-performance model for structured/tabular data.
- Uses boosting to improve accuracy.

### **4️⃣ Support Vector Machine (SVM)**
- Effective for small datasets with clear margins.

### **5️⃣ Neural Networks (MLP)**
- Can capture complex patterns in data.
- Requires more computational resources.

### **6️⃣ Model Comparison**
All models are evaluated based on:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **ROC-AUC**
- **Learning Curves**
- **Confusion Matrix**

---

## **API Integration**
This project integrates with **DeepSeek AI API** for **AI-enhanced treatment recommendations**.  
**API Endpoint:** `https://api.siliconflow.cn/v1/chat/completions`  

### **How API Works**
1. **Patient data** is sent to the API.
2. **AI generates personalized treatment recommendations**.
3. **Recommendations are displayed in the web app**.
4. **If API fails, fallback recommendations are used**.

---

## **Results and Visualizations**
### **1️⃣ Feature Importance**
[Feature Importance](results/feature_importance_rf.png)

### **2️⃣ Model Performance Comparison**
[Model Comparison](results/model_comparison.png)

### **3️⃣ Confusion Matrix**
[Confusion Matrix](results/Random_Forest_confusion_matrix.png)

### **4️⃣ ROC Curve**
[ROC Curve](results/Random_Forest_roc_curve.png)

---

## **Contributors**
👨‍💻 **Team Members**  
- **ZHU Jun** 
- **[ZHAO Tailai]**  
- **[ZHAO Yuhao]**  
- **[ZHANG Tianze]** 
- **[LAW Hoi]**
- **[OU Lefeng]**
- **[CHEN Jiaqi]** 

---

## **License**
📜 **MIT License**  
This project is open-source and available for free use under the MIT License.
