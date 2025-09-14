## Bitcoin Price Prediction Web App 📈🤖
A Streamlit-based web application for predicting Bitcoin price movements using machine learning models! 🚀

## ✨ Features
📊 Interactive Data Visualization: Dynamic charts showing Bitcoin price trends, distributions, and correlations

🤖 Multiple ML Models: Support for Logistic Regression, SVM, and XGBoost classifiers

📈 Performance Metrics: AUC scores for training and validation sets with detailed visualizations

🔮 Next-Day Prediction: Predicts whether Bitcoin price will increase or decrease the next day

🎨 User-Friendly Interface: Clean, intuitive design with interactive components

⚡ Real-time Updates: Live data fetching capabilities (when available)

## 🛠️ Installation
Clone this repository:
```
bash
git clone <your-repo-url>
cd bitcoin-price-prediction
Create a virtual environment (highly recommended):
```
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install the required dependencies:
```
```bash
pip install -r requirements.txt
🚀 Usage
Run the Streamlit application:
```
```bash
streamlit run bitcoin_prediction_app.py
Open your web browser and navigate to the local URL shown in the terminal (typically http://localhost:8501)
```
Use the sidebar to:

📁 Upload your own CSV file with Bitcoin price data

⚙️ Configure data preprocessing options

🤖 Select a machine learning model

📊 Explore the visualizations and model performance metrics

# 📋 Data Format
The application expects CSV files with the following columns:

Date 📅

Open 🔓

High 📈

Low 📉

Close 🔒

Adj Close (optional) 🔧

# Volume 💹

If no file is uploaded, the application will generate sample data for demonstration purposes.

## 🤖 Model Details
The application implements three powerful machine learning models:

📊 Logistic Regression: A linear model for classification

🧠 Support Vector Machine (SVM): With polynomial kernel

⚡ XGBoost: A gradient boosting algorithm known for its performance

Features used for prediction:

Open-Close difference

Low-High difference

Quarter-end indicator

Technical indicators (when available)

## 📁 Project Structure
text<br>
bitcoin-price-prediction/<br>
├── app.py  # 🎯 Main application file<br>
├── requirements.txt           # 📦 Python dependencies<br>
├── bitcoin.csv/              # 💾 Sample datasets (optional)<br>
├── assets/                   # 🎨 Images and icons (optional)<br>
└── README.md                 # 📖 This file<br> 

## 📦 Dependencies
numpy==1.26.4

pandas==2.2.3

matplotlib==3.9.2

seaborn==0.13.2

streamlit==1.38.0

scikit-learn==1.5.1

xgboost==2.1.1

## ⚠️ Disclaimer
🚨 This application is for educational purposes only! 🚨

Cryptocurrency investments carry significant risk, and predictions made by this tool should not be considered financial advice. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.

## 📜 License
This project is open source and available under the MIT License.

Made with ❤️ and Python 🐍


Feel free to contribute by submitting issues or pull requests! 🤝

