# 👨‍💼 Employee Turnover Prediction

Machine learning web application to predict whether an employee is likely to leave (resign) or stay in a company, built with **Python**, **Scikit-learn**, and **Streamlit**.

## 📌 Overview

This project uses a **Random Forest Classifier** trained on employee performance and productivity data to predict employee turnover risk. Based on details like age, department, salary, overtime hours, satisfaction score, and performance score, the model predicts whether an employee is likely to **stay** or **leave**, along with a confidence score and risk level.

## 🚀 Features

- 🎯 Real-time employee turnover prediction
- 📊 Prediction confidence score with risk categorization (Low / Medium / High)
- 📈 Visual insights — prediction probability pie chart & feature importance chart
- 📋 Employee summary and personalized recommendations
- 📥 Downloadable prediction report (CSV)
- 🖥️ Clean and interactive Streamlit UI
- ✅ Model accuracy: **98.7%**

## 🖼️ Screenshot

### Input Form & Prediction
![App Form](images/app_form.png)

### Employee Summary & History
![Prediction Result](images/prediction_result.png)

### Prediction Insights & Feature Importance
![Feature Importance](images/feature_importance.png)

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Scikit-learn | Model building (Random Forest Classifier) |
| Streamlit | Web application interface |
| Pandas | Data handling and preprocessing |
| Matplotlib | Data visualization (charts) |
| Joblib | Model saving/loading |

## 📊 Model Details

- **Algorithm:** Random Forest Classifier (`n_estimators=200`)
- **Accuracy:** 98.7%
- **Preprocessing:** Median imputation for numeric features, most-frequent imputation + One-Hot Encoding for categorical features

### Input Features

| Feature | Description |
|---|---|
| Department | Employee's department (HR, Sales, IT, Finance, etc.) |
| Gender | Male / Female |
| Age | Employee's age |
| Years at Company | Number of years employed |
| Monthly Salary | Employee's monthly salary |
| Overtime Hours | Overtime hours worked |
| Employee Satisfaction Score | Score from 1–10 |
| Performance Score | Score from 1–5 |

## 📂 Project Structure
Employee_Turnover_prediction/
│
├── app/
│   └── app.py                # Main Streamlit application
├── Data/
│   └── Extended_Employee_Performance_and_Productivity_Data.csv
├── model/
│   └── random_forest_model.pkl
├── Notebooks/                # Jupyter notebooks (EDA, training)
├── images/                   # Screenshots for README
├── requirements.txt
├── LICENSE
└── README.md


## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/rahul-dev48/Employee-Turnover-prediction.git
cd Employee-Turnover-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## ▶️ How to Run

```bash
streamlit run app/app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`) in your browser.

## 📈 How It Works

1. Enter employee details (age, department, salary, overtime hours, satisfaction & performance scores) in the app
2. Click **Predict Employee Turnover**
3. The model predicts whether the employee is likely to **Stay** or **Leave**
4. View prediction confidence, risk level, personalized recommendations, and visual insights
5. Download the prediction report as CSV

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 👤 Author By

**Rahul Kumar**
GitHub: [@rahul-dev48](https://github.com/rahul-dev48)
