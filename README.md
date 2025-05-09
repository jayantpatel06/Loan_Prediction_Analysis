🏦 Loan Prediction Analysis
This project provides an end-to-end pipeline for predicting loan approval status using machine learning models. The workflow includes data preprocessing, exploratory data analysis (EDA), feature engineering, model building, evaluation, and performance visualization.

📂 Dataset
The dataset is sourced from the Loan Prediction Problem and contains applicant details such as income, employment status, credit history, loan amount, and more.

🔧 Features
Data Cleaning & Preprocessing

Handled missing values using mean (numerical) and mode (categorical).

Performed log transformation to normalize skewed distributions.

Dropped irrelevant or redundant columns.

Feature Engineering

Created Total_Income by combining applicant and co-applicant income.

Applied log transformations to skewed features.

Label Encoding

Converted categorical variables to numerical using LabelEncoder.

Exploratory Data Analysis (EDA)

Visualized categorical distributions using count plots.

Assessed numerical features using histograms and KDE plots.

Examined feature correlations using a heatmap.

Model Training & Evaluation

Trained and compared three models:

Logistic Regression

Decision Tree

Random Forest

Evaluated using:

Accuracy

AUC (Area Under ROC Curve)

Mean Squared Error (MSE)

Cross-validation score

Visualized ROC curves and confusion matrices.

🧪 Models & Evaluation
Each model was trained using train_test_split and evaluated on:

Test Accuracy

Mean Squared Error

AUC Score

Cross-Validation Accuracy

ROC curves and confusion matrices were plotted to provide deeper insight into model performance and classification balance.

📈 Visualizations
Count plots for categorical features

Histograms for numerical/log-transformed features

Correlation heatmap

ROC Curve comparison

Confusion matrices for each classifier

📁 File Structure

loan-prediction-analysis/
│
├── loan_prediction.ipynb        # Main analysis notebook
├── README.md                    # Project overview and documentation
└── requirements.txt             # (Optional) Python dependencies

🚀 How to Run
Clone the repository:

git clone https://github.com/your-username/loan-prediction-analysis.git

cd loan-prediction-analysis

Install required packages (if using virtualenv or conda):

pip install -r requirements.txt

Run the notebook:

Open loan_prediction.ipynb in Jupyter Notebook or any compatible IDE.


🛠 Technologies Used
Python 3.x

pandas, numpy

matplotlib, seaborn

scikit-learn

📌 Future Improvements
Hyperparameter tuning with GridSearchCV

Feature selection using recursive feature elimination (RFE)

Deployment as a Flask or Streamlit web app

Model interpretability with SHAP or LIME

📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.

🤝 Contributing
Pull requests and issues are welcome. For major changes, please open a discussion first.
