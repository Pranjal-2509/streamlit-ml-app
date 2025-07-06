import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.express as px

# Loading the dataset + caching
@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data['data'], columns=data['feature_names'])
    df['target'] = data['target']  # 0 = malignant, 1 = benign
    return df, data['feature_names']

# Training the model + caching
@st.cache_resource
def train_model(df):
    X = df[feature_cols]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def main():
    st.title("🩷Breast Cancer Predictor🩷")
    st.write("Enter tumor measurements to predict benign vs. malignant")

    # Load data and model
    df, all_features = load_data()
    global feature_cols
    # pick a few interpretable features
    feature_cols = ['mean radius', 'mean texture', 'mean smoothness']
    model, X_test, y_test = train_model(df)

    # Sidebar inputs
    st.sidebar.header("🔧 Input Features")
    inputs = {}
    for feat in feature_cols:
        low, high = float(df[feat].min()), float(df[feat].max())
        mean = float(df[feat].mean())
        inputs[feat] = st.sidebar.slider(
            feat, min_value=low, max_value=high, value=mean
        )

    if st.button("Predict Diagnosis"):
        x = np.array([list(inputs.values())])
        prob = model.predict_proba(x)[0,1]
        pred = model.predict(x)[0]

        label = "Benign ✅" if pred==1 else "Malignant ⚠️"
        st.metric("Prediction", label, delta=f"{prob:.2%} benign likelihood")

        # Show model accuracy
        acc = model.score(X_test, y_test)
        st.write(f"Model accuracy on hold-out set: **{acc:.2%}**")

        # Plot distribution of one feature colored by diagnosis
        fig = px.histogram(
            df, x='mean radius', color='target',
            barmode='overlay',
            labels={'mean radius':'Mean Radius','target':'Diagnosis'},
            color_discrete_map={0:'red',1:'green'},
            title="Mean Radius Distribution (0=Malignant, 1=Benign)"
        )
        # mark your input
        fig.add_scatter(
            x=[inputs['mean radius']], y=[0],
            mode='markers+text',
            marker=dict(size=15, symbol='x'),
            text=["You"], textposition="top center",
            name="Your Tumor"
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
