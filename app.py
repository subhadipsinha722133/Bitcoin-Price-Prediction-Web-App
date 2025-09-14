import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Bitcoin Price Prediction",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.sidebar.header("Made by Subhadip 🧑‍💻")

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF9900;
        text-align: center;
    }
    .prediction-positive {
        color: #00CC00;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .prediction-negative {
        color: #FF0000;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">₿ Bitcoin Price Prediction</h1>', unsafe_allow_html=True)


st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:

    st.sidebar.info("Using demo data. Upload a CSV file to use your own data.")

    dates = pd.date_range(start='2018-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    base_price = 5000
    price_data = []
    
    for i in range(len(dates)):
        volatility = np.random.normal(0, 0.02)
        if i > 0:
            price = price_data[i-1]['Close'] * (1 + volatility)
        else:
            price = base_price * (1 + volatility)
            
        open_price = price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, price) * (1 - abs(np.random.normal(0, 0.01)))
        close_price = price
        volume = np.random.lognormal(15, 1)
        
        price_data.append({
            'Date': dates[i].strftime('%Y-%m-%d'),
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume
        })
    
    df = pd.DataFrame(price_data)
    df['Adj Close'] = df['Close']  # Adding Adj Close for compatibility

st.sidebar.header("Data Preprocessing")
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.dataframe(df.head())

duplicates = df.duplicated().sum()
if duplicates > 0:
    st.sidebar.warning(f"Dataset contains {duplicates} duplicates. Consider cleaning the data.")

if 'Adj Close' in df.columns:
    if df[df['Close'] == df['Adj Close']].shape[0] == df.shape[0]:
        df = df.drop(['Adj Close'], axis=1)

df['Date'] = pd.to_datetime(df['Date'])

splitted = df['Date'].dt.strftime('%Y-%m-%d').str.split('-', expand=True)
df['year'] = splitted[0].astype('int')
df['month'] = splitted[1].astype('int')
df['day'] = splitted[2].astype('int')
df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

st.subheader("Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Number of Samples", df.shape[0])
col2.metric("Number of Features", df.shape[1])
col3.metric("Date Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")


st.subheader("Data Visualization")


fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df['Date'], df['Close'], color='orange')
ax.set_title('Bitcoin Closing Price Over Time', fontsize=15)
ax.set_ylabel('Price in USD')
ax.grid(True)
st.pyplot(fig)


st.write("Distribution of Price Features")
features = ['Open', 'High', 'Low', 'Close']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for i, col in enumerate(features):
    row, col_idx = i // 2, i % 2
    sns.histplot(df[col], kde=True, ax=axes[row, col_idx])
    axes[row, col_idx].set_title(f'Distribution of {col}')
plt.tight_layout()
st.pyplot(fig)

st.write("Box Plots of Price Features")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for i, col in enumerate(features):
    row, col_idx = i // 2, i % 2
    sns.boxplot(y=df[col], ax=axes[row, col_idx], orient='v')
    axes[row, col_idx].set_title(f'Box Plot of {col}')
plt.tight_layout()
st.pyplot(fig)

st.write("Yearly Average Prices")
data_grouped = df.groupby('year').mean()
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
    row, col_idx = i // 2, i % 2
    data_grouped[col].plot.bar(ax=axes[row, col_idx], color='orange')
    axes[row, col_idx].set_title(f'Average {col} by Year')
    axes[row, col_idx].tick_params(axis='x', rotation=45)
plt.tight_layout()
st.pyplot(fig)
st.write("Target Variable Distribution")
fig, ax = plt.subplots(figsize=(6, 6))
target_counts = df['target'].value_counts()
ax.pie(target_counts.values, labels=['Down (0)', 'Up (1)'], autopct='%1.1f%%', 
       colors=['#ff4b4b', '#4bff4b'], startangle=90)
ax.set_title('Distribution of Price Movement Predictions')
st.pyplot(fig)

# Correlation heatmap
st.write("Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
            square=True, ax=ax, cbar_kws={"shrink": .8})
ax.set_title("Feature Correlation Matrix")
st.pyplot(fig)


st.subheader("Model Training")

features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

valid_indices = target.notna()
features = features[valid_indices]
target = target[valid_indices]

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
    features_scaled, target, test_size=0.3, random_state=42, shuffle=False
)

models = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(kernel='poly', probability=True),
    "XGBoost": XGBClassifier()
}

selected_model = st.selectbox("Select Model", list(models.keys()))

if st.button("Train Model"):
    with st.spinner('Training model...'):
        model = models[selected_model]
        model.fit(X_train, Y_train)
        
       
        train_accuracy = metrics.roc_auc_score(Y_train, model.predict_proba(X_train)[:, 1])
        valid_accuracy = metrics.roc_auc_score(Y_valid, model.predict_proba(X_valid)[:, 1])
    
        col1, col2 = st.columns(2)
        col1.markdown('<div class="metric-card">', unsafe_allow_html=True)
        col1.metric("Training AUC Score", f"{train_accuracy:.4f}")
        col1.markdown('</div>', unsafe_allow_html=True)
        
        col2.markdown('<div class="metric-card">', unsafe_allow_html=True)
        col2.metric("Validation AUC Score", f"{valid_accuracy:.4f}")
        col2.markdown('</div>', unsafe_allow_html=True)
        
        # Confusion matrix
        st.write("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay.from_estimator(model, X_valid, Y_valid, ax=ax, cmap='Blues')
        ax.set_title(f'Confusion Matrix - {selected_model}')
        st.pyplot(fig)
        
        # Make prediction for the next day
        latest_data = features_scaled[-1].reshape(1, -1)
        prediction_proba = model.predict_proba(latest_data)[0]
        prediction = model.predict(latest_data)[0]
        
        st.subheader("Next Day Prediction")
        if prediction == 1:
            st.markdown(f'<p class="prediction-positive">Prediction: Price will likely INCREASE (confidence: {prediction_proba[1]:.2%})</p>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="prediction-negative">Prediction: Price will likely DECREASE (confidence: {prediction_proba[0]:.2%})</p>', 
                       unsafe_allow_html=True)

# Show feature importance for tree-based models
if selected_model == "XGBoost" and st.checkbox("Show Feature Importance"):
    model = XGBClassifier()
    model.fit(X_train, Y_train)
    
    feature_importance = model.feature_importances_
    feature_names = ['Open-Close', 'Low-High', 'Is Quarter End']
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color='orange')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance for XGBoost Model')
    st.pyplot(fig)
    
if st.sidebar.checkbox("Show processed data"):
    st.subheader("Processed Data with Features")
    st.dataframe(df.tail(10))

st.markdown("---")
st.markdown("""
**Note:** This prediction is based on historical data and machine learning models. 
Cryptocurrency markets are highly volatile and unpredictable. 
This tool should be used for educational purposes only, not for financial advice.
""")