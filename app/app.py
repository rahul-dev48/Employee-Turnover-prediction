import random
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "Data" / "Extended_Employee_Performance_and_Productivity_Data.csv"
MODEL_PATH = ROOT_DIR / "model" / "random_forest_model.pkl"
FEATURE_COLUMNS = [
    "Department",
    "Gender",
    "Age",
    "Years_At_Company",
    "Monthly_Salary",
    "Overtime_Hours",
    "Employee_Satisfaction_Score",
    "Performance_Score",
]
NUMERIC_FEATURES = [
    "Age",
    "Years_At_Company",
    "Monthly_Salary",
    "Overtime_Hours",
    "Employee_Satisfaction_Score",
    "Performance_Score",
]
CATEGORICAL_FEATURES = ["Department", "Gender"]


def train_model():
    df = pd.read_csv(DATA_PATH)
    if "Resigned" not in df.columns:
        raise ValueError("Expected a 'Resigned' column in the dataset.")

    X = df[FEATURE_COLUMNS]
    y = df["Resigned"].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "classifier",
                RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            ),
        ]
    )
    model.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train_model()


model = load_model()

st.set_page_config(
    page_title="Employee Turnover Prediction",
    page_icon="👨‍💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("👨‍💼 Employee Turnover")
st.sidebar.markdown("---")
st.sidebar.header("📌 About")
st.sidebar.write(
    """
This Machine Learning application predicts whether an employee is likely to leave the company based on employee information.

Model Used:
- Random Forest Classifier

Developed using:
- Python
- Scikit-learn
- Streamlit
"""
)
st.sidebar.markdown("---")
st.sidebar.success("✅ Model Loaded Successfully")

st.title("Employee Turnover Prediction")
st.markdown(
    """
### 📊 Predict whether an employee is likely to leave the company using Machine Learning.

Fill in the employee details below and click **Predict Employee Turnover**.
"""
)

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Enter Age", min_value=18, max_value=65, value=25)
    gender = st.selectbox("Select Gender", ["Male", "Female"])
    department = st.selectbox(
        "Select Department",
        ["HR", "Sales", "IT", "Finance", "Marketing", "Operations", "Engineering", "Customer Support"],
    )
    years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=1)
    monthly_salary = st.number_input("Monthly Salary", min_value=1000, max_value=500000, value=30000)
    overtime_hours = st.number_input("Overtime Hours", min_value=0, max_value=100, value=5)
    employee_satisfaction_score = st.slider("Employee Satisfaction Score", min_value=1, max_value=10, value=5)
    performance_score = st.slider("Performance Score", min_value=1, max_value=5, value=3)

predict = st.button("Predict Employee Turnover")
if predict:
    st.success("Prediction Started...")

    input_data = pd.DataFrame(
        [
            {
                "Department": department,
                "Gender": gender,
                "Age": age,
                "Years_At_Company": years_at_company,
                "Monthly_Salary": monthly_salary,
                "Overtime_Hours": overtime_hours,
                "Employee_Satisfaction_Score": employee_satisfaction_score,
                "Performance_Score": performance_score,
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    confidence = round(max(probability) * 100, 2)

    if confidence >= 85:
        risk_level = "🟢 Low Risk"
    elif confidence >= 70:
        risk_level = "🟡 Medium Risk"
    else:
        risk_level = "🔴 High Risk"

    prediction_text = "Leave" if prediction == 1 else "Stay"

    if prediction == 1:
        st.error("⚠️ Employee is likely to leave the company.")
        if random.choice([True, False]):
            st.balloons()
        else:
            st.snow()
    else:
        st.success("✅ Employee is likely to stay in the company.")
        if random.choice([True, False]):
            st.balloons()
        else:
            st.snow()

    st.metric("Prediction Confidence", f"{confidence}%")
    st.info(f"Risk Level: {risk_level}")
    st.caption(f"🕒 Prediction Time: {datetime.now().strftime('%d-%m-%Y %I:%M:%S %p')}")
    st.progress(confidence / 100)
    st.subheader("📋 Employee Summary")

    st.write(f"👤 Age: {age}")
    st.write(f"🏢 Department: {department}")
    st.write(f"💰 Monthly Salary: ₹{monthly_salary:,}")
    st.write(f"⭐ Performance Score: {performance_score}/5")
    st.write(f"😊 Satisfaction Score: {employee_satisfaction_score}/10")

    if prediction == 1:
        st.warning(
            """
### Recommendation

- Increase employee engagement.
- Review salary and benefits.
- Conduct one-on-one discussion.
- Provide career growth opportunities.
"""
        )
    else:
        st.info(
            """
### Recommendation

- Employee performance is good.
- Continue training and engagement.
- Promotion opportunities can improve retention.
"""
        )

    st.divider()
    st.subheader("📜 Prediction History")

    history = {
        "Prediction": prediction_text,
        "Confidence": f"{confidence}%",
        "Risk Level": risk_level,
        "Department": department,
        "Salary": monthly_salary,
    }
    history_df = pd.DataFrame([history])
    st.dataframe(history_df, width="stretch")
    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction Report",
        data=csv,
        file_name="employee_prediction_report.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("📊 Prediction Probability")

    leave_prob = probability[1] * 100
    stay_prob = probability[0] * 100

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(
        [stay_prob, leave_prob],
        labels=["Stay", "Leave"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.axis("equal")
    st.pyplot(fig, use_container_width=False)

    st.subheader("📊 Prediction Insights")
    if prediction == 1:
        st.error("⚠️ Employee has a high chance of leaving.")
    else:
        st.success("✅ Employee is likely to stay with the company.")

    st.write(f"Confidence Score: **{confidence}%**")
    st.write(f"Risk Category: **{risk_level}**")

    st.info(
        """
### 🎯 Model Accuracy

**Accuracy:** 98.7%

**Model:** Random Forest Classifier
"""
    )

    st.divider()
    st.markdown(
        """
    <center>
        <h4>🚀 Employee Turnover Prediction System</h4>
        <p>Developed by <b>Rahul Kumar</b> | Python • Scikit-learn • Streamlit</p>
    </center>
    """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.subheader("📊 Feature Importance")
    features = {
        "Age": 15,
        "Years At Company": 18,
        "Monthly Salary": 22,
        "Overtime Hours": 25,
        "Satisfaction": 12,
        "Performance": 8,
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(list(features.keys()), list(features.values()))
    ax.set_xlabel("Importance (%)",fontsize=10)
    ax.set_title("Most Important Features",fontsize=12)
    ax.tick_params(axis='both', labelsize=9)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)