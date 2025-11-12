🏋️‍♂️ Athletic Performance Prediction (ML + Streamlit)
📋 Project Overview

This project predicts an athlete’s performance level (Beginner, Intermediate, Advanced, or Elite) 
using a Machine Learning Classification Model built with scikit-learn and deployed via Streamlit.

A synthetic dataset was generated to simulate real-world physiological and training data, making the project fully reproducible and deployable anywhere.

🎯 Objective

To demonstrate the use of supervised machine learning for predicting human athletic performance based on measurable attributes like fitness, endurance, and training patterns.

⚙️ Tech Stack
Category	Technologies Used
Programming	Python
Data Handling	Pandas, NumPy
Machine Learning	Scikit-learn
Model Persistence	Joblib
Web App Framework	Streamlit
Version Control	GitHub
🧩 Features

✅ Predicts categorical performance levels (Beginner / Intermediate / Advanced / Elite)
✅ Auto-generated synthetic dataset — no external data required
✅ Built with Random Forest Classifier (robust and high accuracy)
✅ Real-time prediction interface using Streamlit Web App
✅ Deployed easily on Streamlit Cloud
✅ Trained model accuracy: 88.4%

📊 Dataset Description

The dataset (athlete_data.csv) includes 2,500 samples with the following features:

Feature	Description
Age	Athlete’s age
Gender	Male / Female
Height (cm)	Body height
Weight (kg)	Body weight
BMI	Calculated from height and weight
Training Hours/Week	Total training hours per week
Resting Heart Rate	Average resting heart rate (BPM)
VO₂ Max	Aerobic endurance indicator
Reaction Time (ms)	Average reaction speed
Sleep Hours	Average daily sleep duration
Performance Level	Target label (Beginner / Intermediate / Advanced / Elite)

🧠 Model Details

Algorithm: RandomForestClassifier

Scaler: StandardScaler

Encoder: LabelEncoder

Evaluation Metric: Accuracy Score

Achieved Accuracy: 88.4%

The model captures relationships between physiological, lifestyle, and training features to classify overall athletic performance.

📂 Project Structure
Athletic_Performance_Prediction/
│
├── app.py                         # Streamlit app
├── athlete_model_training.ipynb    # Model training notebook
├── athlete_model_training.py       # Python script version
├── athlete_model.pkl               # Trained model
├── scaler.pkl                      # Input feature scaler
├── label_encoder.pkl               # Encodes target labels
├── synthetic_athlete_data.csv      # Synthetic dataset
├── requirements.txt                # Dependencies
├── README.txt                       # Documentation

🧑‍💻 How to Run Locally
Step 1: Clone the Repository
git clone https://github.com/soojalkumar337/Athletic_Performance_Prediction.git
cd Athletic_Performance_Prediction

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Run Streamlit App
streamlit run app.py

🌟 Example Use Cases

Predicting potential athletic performance for training programs

Fitness tracking & improvement recommendation apps

Sports data analytics demonstrations

Machine learning portfolio showcase

🏆 Future Enhancements

Integrate advanced algorithms (XGBoost, LightGBM)

Add feature importance visualization

Incorporate user-uploaded CSV prediction

Connect to wearable fitness device APIs for live data

## 🌐 Live Demo
[👉 Try the App Here](https://athleticperformanceprediction-zflfeong3zpdvhpcbpkaux.streamlit.app)


📧 Developed by:

Name: Soojal Kumar
Email: kumarsoojal55@gmail.com