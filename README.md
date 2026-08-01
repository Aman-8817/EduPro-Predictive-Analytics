# EduPro Predictive Intelligence Dashboard

## 🚀 Live Dashboard

[Open EduPro Predictive Analytics Dashboard](https://aman-edupro-predictive-analytics.streamlit.app/)

## Project Overview

This Unified Mentor Data Analyst Internship project converts EduPro's historical course data into month-ahead forecasts for:

- course enrollment demand;
- revenue per course; and
- aggregated revenue by course category.

The revised implementation uses a time-based holdout so future months are not mixed into training data.

## Source Data

- Users: 3,000 records
- Courses: 60 records
- Teachers: 60 records
- Transactions: 10,000 records

## Feature Engineering

- Price bands: Free, Low, High
- Duration buckets: Short, Medium, Long
- Rating tiers: Developing, Good, Excellent
- Course type, level, and category encoding
- Instructor-experience buckets and teacher rating
- Expertise-category match score
- Past enrollment count
- Historical average revenue
- Revenue per enrollment

## Models Compared

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Evaluation uses MAE, RMSE, and R² on November–December 2025, which are held out as future forecast months.

## Dashboard Modules

- Home and business problem
- Historical Insights
- Course Demand and Revenue Forecast
- Category-Level Revenue Forecast
- Model Performance and Feature Importance
- About and methodology

## Run the Project

1. Open a terminal in this folder.
2. Install packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Regenerate modeling artifacts if required:

   ```bash
   python train_models.py
   ```

4. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

## Main Files

- `EduPro_Forecasting_Final.ipynb`: final explanatory notebook
- `train_models.py`: reproducible feature engineering and model training
- `app.py`: Streamlit dashboard
- `model_bundle.joblib`: three fitted deployment pipelines
- `model_metrics.csv`: model comparison results
- `feature_importance.csv`: top model drivers
- `course_month_forecasting.csv`: course-month modeling dataset
- `category_month_forecasting.csv`: category-month modeling dataset

## Developer

**Aman Jat**  
Unified Mentor Data Analyst Internship
