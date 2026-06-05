import streamlit as st
import pandas as pd
import numpy as np
import pickle

import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

[data-testid="stMetric"]{
    background:white;
    padding:15px;
    border-radius:15px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.1);
}

h1,h2,h3{
    color:#0F172A;
}

.stButton button{
    width:100%;
    border-radius:10px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# LOAD DATA
# -----------------------------------

df = pd.read_csv("insurance.csv")

# -----------------------------------
# LOAD MODEL
# -----------------------------------

saved = pickle.load(open("model.pkl","rb"))

model = saved["model"]

sex_encoder = saved["sex_encoder"]
smoker_encoder = saved["smoker_encoder"]
region_encoder = saved["region_encoder"]

# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("🏥 Insurance App")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dashboard",
        "🔍 Prediction",
        "⭐ Feature Importance",
        "📈 Model Performance"
    ]
)

# ===================================
# HOME PAGE
# ===================================

if page=="🏠 Home":

    st.title("🏥 Medical Insurance Analytics")

    st.markdown(
    """
    Predict medical insurance charges using Machine Learning.
    """
    )

    col1,col2,col3,col4 = st.columns(4)

    col1.metric(
        "Total Records",
        len(df)
    )

    col2.metric(
        "Average Charges",
        f"${df['charges'].mean():,.0f}"
    )

    col3.metric(
        "Average BMI",
        round(df['bmi'].mean(),2)
    )

    col4.metric(
        "Average Age",
        round(df['age'].mean(),0)
    )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.divider()

    st.subheader("Dataset Statistics")

    st.dataframe(df.describe())

# ===================================
# DASHBOARD
# ===================================

elif page=="📊 Dashboard":

    st.title("📊 Exploratory Data Analysis")

    col1,col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="charges",
            nbins=30,
            title="Charges Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.box(
            df,
            x="smoker",
            y="charges",
            color="smoker",
            title="Charges by Smoker"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    col3,col4 = st.columns(2)

    with col3:

        fig = px.scatter(
            df,
            x="age",
            y="charges",
            color="smoker",
            size="bmi",
            title="Age vs Charges"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col4:

        fig = px.scatter(
            df,
            x="bmi",
            y="charges",
            color="smoker",
            title="BMI vs Charges"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    fig = px.bar(
        df.groupby("region")["charges"]
        .mean()
        .reset_index(),
        x="region",
        y="charges",
        title="Average Charges by Region"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ===================================
# PREDICTION PAGE
# ===================================

elif page=="🔍 Prediction":

    st.title("🔍 Predict Insurance Charges")

    col1,col2 = st.columns(2)

    with col1:

        age = st.slider(
            "Age",
            18,
            65,
            25
        )

        sex = st.selectbox(
            "Gender",
            ["male","female"]
        )

        bmi = st.slider(
            "BMI",
            15.0,
            50.0,
            25.0
        )

    with col2:

        children = st.selectbox(
            "Children",
            [0,1,2,3,4,5]
        )

        smoker = st.selectbox(
            "Smoker",
            ["yes","no"]
        )

        region = st.selectbox(
            "Region",
            [
                "northeast",
                "northwest",
                "southeast",
                "southwest"
            ]
        )

    if st.button("Predict Charges"):

        sex_val = sex_encoder.transform([sex])[0]
        smoker_val = smoker_encoder.transform([smoker])[0]
        region_val = region_encoder.transform([region])[0]

        input_df = pd.DataFrame({

            "age":[age],
            "sex":[sex_val],
            "bmi":[bmi],
            "children":[children],
            "smoker":[smoker_val],
            "region":[region_val]

        })

        prediction = model.predict(input_df)[0]

        st.success(
            f"Predicted Charges = ${prediction:,.2f}"
        )

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={'text':"Predicted Charges"},
            gauge={
                'axis':{
                    'range':[0,70000]
                }
            }
        ))

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ===================================
# FEATURE IMPORTANCE
# ===================================

elif page=="⭐ Feature Importance":

    st.title("⭐ Feature Importance")

    try:

        importance = model.feature_importances_

        features = [
            "age",
            "sex",
            "bmi",
            "children",
            "smoker",
            "region"
        ]

        imp_df = pd.DataFrame({

            "Feature":features,
            "Importance":importance

        })

        imp_df = imp_df.sort_values(
            by="Importance",
            ascending=False
        )

        fig = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(imp_df)

    except:

        st.warning(
            "Feature importance unavailable."
        )

# ===================================
# MODEL PERFORMANCE
# ===================================

elif page=="📈 Model Performance":

    st.title("📈 Model Performance")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        r2_score,
        mean_absolute_error,
        mean_squared_error
    )

    data = df.copy()

    data["sex"] = sex_encoder.transform(data["sex"])
    data["smoker"] = smoker_encoder.transform(data["smoker"])
    data["region"] = region_encoder.transform(data["region"])

    X = data.drop("charges",axis=1)
    y = data["charges"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "R² Score",
        round(r2,3)
    )

    c2.metric(
        "MAE",
        round(mae,2)
    )

    c3.metric(
        "RMSE",
        round(rmse,2)
    )

    result_df = pd.DataFrame({
        "Actual":y_test,
        "Predicted":y_pred
    })

    fig = px.scatter(
        result_df,
        x="Actual",
        y="Predicted",
        title="Actual vs Predicted"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )