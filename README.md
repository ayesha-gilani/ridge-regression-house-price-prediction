
# 🏠 Ames House Price Prediction Pipeline (Ridge Regression)

An end-to-end supervised machine learning regression pipeline built with Python and Scikit-Learn to predict residential property sale prices using the Ames Housing dataset.

---

## 📌 Project Overview

Predicting real estate prices requires handling mixed data types, missing values, and high-dimensional categorical features without introducing data leakage.

This project implements a structured `ColumnTransformer` preprocessing workflow combined with **Ridge Regression (L2 regularization)** to improve model generalization across 302 encoded features.

The objective is to develop a complete machine learning pipeline covering data ingestion, preprocessing, model training, evaluation, and visualization.

---

## 📊 Dataset Summary

- **Dataset:** Ames Housing Dataset
- **Source:** OpenML
- **Loading Method:** `fetch_openml(name="house_prices", version=1)`
- **Dimensions:** 1,460 rows × 81 columns
- **Input Features:** 79 (after dropping `Id`)
- **Target Variable:** `SalePrice`

### Target Variable Statistics

| Statistic | Value |
|---|---:|
| Mean Sale Price | $180,921.20 |
| Median Sale Price | $163,000.00 |
| Minimum Sale Price | $34,900.00 |
| Maximum Sale Price | $755,000.00 |

### Missing Values

- Total missing entries: 6,965
- Numerical columns with missing values: 3
- Categorical columns with missing values: 16
- Total affected columns: 19

---

## ⚙️ Machine Learning Workflow

### 1. Data Ingestion and Exploration
- Loaded the Ames Housing dataset using Scikit-Learn's OpenML interface.
- Explored dataset dimensions, data types, descriptive statistics, and missing values.
- Separated input features (`X`) and target variable (`y`).
- Dropped the unnecessary `Id` column.

### 2. Train-Test Splitting

Split the dataset into training and testing subsets:

- Training data: 80% (1,168 samples)
- Testing data: 20% (292 samples)
- Random state: `42`

### 3. Data Preprocessing

Implemented a `ColumnTransformer` with separate preprocessing pipelines.

**Numerical Pipeline (36 Features)**

- Missing value imputation using `SimpleImputer(strategy='mean')`.
- Feature scaling using `StandardScaler`.

**Categorical Pipeline (43 Features)**

- Missing value imputation using `SimpleImputer(strategy='constant', fill_value='Missing')`.
- Categorical encoding using `OneHotEncoder(handle_unknown='ignore')`.

**Feature Transformation**

The preprocessing pipeline transformed 79 original input columns into 302 numerical features.

### 4. Model Training and Comparison

Implemented and compared the following regression models:

- **Linear Regression:** Baseline regression model.
- **Ridge Regression:** Regularized regression using L2 regularization.
- **RidgeCV:** Cross-validation-based selection of the regularization parameter.

Tested Ridge regularization strengths using:

`alphas = [0.1, 1.0, 5.0, 10.0, 15.0, 20.0, 50.0, 100.0]`

### 5. Model Evaluation

Evaluated model performance using:

- **R² Score:** Measures the proportion of variance in house prices explained by the model.
- **Root Mean Squared Error (RMSE):** Measures prediction error in the original target units.

---

## 📈 Model Evaluation Results

| Model | Parameter | R² Score | RMSE |
|---|---|---:|---:|
| Linear Regression | Default | 0.4435 | — |
| Ridge Regression | alpha = 10.0 | **0.8785** | **$30,529.61** |
| RidgeCV | Best alpha = 20.0 | 0.8759 | — |

### Key Findings

- Ridge Regression achieved the best reported test-set performance.
- The model obtained an R² score of **0.8785**, explaining approximately 87.85% of the variation in house sale prices.
- The RMSE was **$30,529.61**.
- Regularization helped improve performance compared with standard Linear Regression.
- RidgeCV selected an alpha value of 20.0 through cross-validation.

---

## 📉 Actual vs. Predicted Visualization

The project includes an Actual vs. Predicted scatter plot to evaluate how closely the model's predictions match actual house prices.

- Points closer to the diagonal reference line indicate more accurate predictions.
- The model showed strong alignment for houses in the $100,000–$300,000 price range.
- Prediction errors showed some variation for higher-priced properties.

---

## 📁 Repository Structure

```text
ridge-regression-house-price-prediction/
│
├── Ridge_Regression_House_Price_Prediction.ipynb
│
├── requirements.txt
│
└── README.md
```

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ridge-regression-house-price-prediction.git

cd ridge-regression-house-price-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook Ridge_Regression_House_Price_Prediction.ipynb
```

**Note:** An active internet connection is required to fetch the dataset from OpenML if it is not already cached.

---

## 🛠️ Tech Stack

- **Python:** Core programming language.
- **Pandas:** Data manipulation and analysis.
- **NumPy:** Numerical computations.
- **Scikit-Learn:** Data preprocessing, pipelines, regression models, cross-validation, and evaluation metrics.
- **Matplotlib:** Data visualization.
- **Seaborn:** Statistical data visualization.

---

## 🔮 Future Improvements

- **Interactive Web Application:** Develop a Streamlit application for interactive house price predictions.
- **Model Benchmarking:** Compare Ridge Regression with Lasso, ElasticNet, Random Forest, and XGBoost.
- **Feature Engineering:** Create additional features such as total square footage and house age to potentially improve predictive performance.
- **Hyperparameter Optimization:** Explore additional regularization strengths and cross-validation strategies.
- **Model Deployment:** Save the trained pipeline and make predictions through a deployable application.

---

## 👩‍💻 Author

**Ayesha Gilani**


*This project demonstrates an end-to-end supervised machine learning regression workflow using Scikit-Learn.*
