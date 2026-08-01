"""Train leakage-safe month-ahead demand and revenue models for EduPro."""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
COURSES_FILE = BASE / "EduPro Online Platform.xlsx - Courses.csv"
TEACHERS_FILE = BASE / "EduPro Online Platform.xlsx - Teachers.csv"
TRANSACTIONS_FILE = BASE / "EduPro Online Platform.xlsx - Transactions.csv"

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "CoursePrice", "CourseDuration", "CourseRating", "YearsOfExperience",
    "TeacherRating", "ExpertiseCategoryMatch", "PastEnrollmentCount",
    "PastAverageRevenue", "RevenuePerEnrollment", "MonthNumber",
]
CATEGORICAL_FEATURES = [
    "CourseType", "CourseLevel", "CourseCategory", "PriceBand",
    "DurationBucket", "RatingTier", "ExperienceBucket",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CATEGORY_NUMERIC = [
    "PastEnrollmentCount", "PastAverageRevenue", "RevenuePerEnrollment",
    "CourseCount", "AverageCoursePrice", "AverageCourseRating", "MonthNumber",
]
CATEGORY_CATEGORICAL = ["CourseCategory"]
CATEGORY_FEATURES = CATEGORY_NUMERIC + CATEGORY_CATEGORICAL


def bucket_features(df):
    out = df.copy()
    out["PriceBand"] = pd.cut(
        out["CoursePrice"], bins=[-1, 0, 250, np.inf],
        labels=["Free", "Low", "High"], include_lowest=True,
    ).astype(str)
    out["DurationBucket"] = pd.cut(
        out["CourseDuration"], bins=[-np.inf, 20, 40, np.inf],
        labels=["Short", "Medium", "Long"],
    ).astype(str)
    out["RatingTier"] = pd.cut(
        out["CourseRating"], bins=[-np.inf, 3.0, 4.0, np.inf],
        labels=["Developing", "Good", "Excellent"],
    ).astype(str)
    out["ExperienceBucket"] = pd.cut(
        out["YearsOfExperience"], bins=[-np.inf, 5, 12, np.inf],
        labels=["Early Career", "Experienced", "Senior"],
    ).astype(str)
    return out


def build_course_month_dataset():
    courses = pd.read_csv(COURSES_FILE)
    teachers = pd.read_csv(TEACHERS_FILE)
    tx = pd.read_csv(TRANSACTIONS_FILE)
    tx["TransactionDate"] = pd.to_datetime(tx["TransactionDate"], dayfirst=True)
    tx["Month"] = tx["TransactionDate"].dt.to_period("M")

    tx_teacher = tx.merge(teachers, on="TeacherID", how="left")
    teacher_stats = tx_teacher.groupby("CourseID", as_index=False).agg(
        YearsOfExperience=("YearsOfExperience", "mean"),
        TeacherRating=("TeacherRating", "mean"),
    )
    match = tx_teacher.merge(courses[["CourseID", "CourseCategory"]], on="CourseID", how="left")
    match["ExpertiseCategoryMatch"] = (
        match["Expertise"].str.casefold() == match["CourseCategory"].str.casefold()
    ).astype(float)
    match_stats = match.groupby("CourseID", as_index=False)["ExpertiseCategoryMatch"].mean()
    course_info = courses.merge(teacher_stats, on="CourseID", how="left").merge(
        match_stats, on="CourseID", how="left"
    )

    months = pd.period_range(tx["Month"].min(), tx["Month"].max(), freq="M")
    grid = pd.MultiIndex.from_product(
        [course_info["CourseID"], months], names=["CourseID", "Month"]
    ).to_frame(index=False)
    monthly = tx.groupby(["CourseID", "Month"], as_index=False).agg(
        CurrentEnrollments=("TransactionID", "count"),
        CurrentRevenue=("Amount", "sum"),
    )
    panel = grid.merge(monthly, on=["CourseID", "Month"], how="left")
    panel[["CurrentEnrollments", "CurrentRevenue"]] = panel[
        ["CurrentEnrollments", "CurrentRevenue"]
    ].fillna(0)
    panel = panel.merge(course_info, on="CourseID", how="left")
    panel = panel.sort_values(["CourseID", "Month"])
    panel["PastEnrollmentCount"] = panel["CurrentEnrollments"]
    panel["PastAverageRevenue"] = panel.groupby("CourseID")["CurrentRevenue"].transform(
        lambda s: s.expanding().mean()
    )
    panel["RevenuePerEnrollment"] = np.where(
        panel["CurrentEnrollments"] > 0,
        panel["CurrentRevenue"] / panel["CurrentEnrollments"], 0,
    )
    panel["TargetEnrollmentCount"] = panel.groupby("CourseID")["CurrentEnrollments"].shift(-1)
    panel["TargetCourseRevenue"] = panel.groupby("CourseID")["CurrentRevenue"].shift(-1)
    panel["ForecastMonth"] = panel["Month"] + 1
    panel["MonthNumber"] = panel["ForecastMonth"].dt.month
    panel = bucket_features(panel)
    panel = panel.dropna(subset=["TargetEnrollmentCount", "TargetCourseRevenue"]).reset_index(drop=True)
    panel["Month"] = panel["Month"].astype(str)
    panel["ForecastMonth"] = panel["ForecastMonth"].astype(str)
    return panel


def build_category_month_dataset(course_panel):
    category = course_panel.groupby(["CourseCategory", "Month", "ForecastMonth"], as_index=False).agg(
        PastEnrollmentCount=("PastEnrollmentCount", "sum"),
        CurrentRevenue=("CurrentRevenue", "sum"),
        PastAverageRevenue=("PastAverageRevenue", "sum"),
        TargetCategoryRevenue=("TargetCourseRevenue", "sum"),
        CourseCount=("CourseID", "nunique"),
        AverageCoursePrice=("CoursePrice", "mean"),
        AverageCourseRating=("CourseRating", "mean"),
        MonthNumber=("MonthNumber", "first"),
    )
    category["RevenuePerEnrollment"] = np.where(
        category["PastEnrollmentCount"] > 0,
        category["CurrentRevenue"] / category["PastEnrollmentCount"], 0,
    )
    return category


def preprocessor(numeric, categorical):
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])


def candidate_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.05, max_iter=20000),
        "Random Forest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.035, max_depth=2,
            loss="huber", random_state=RANDOM_STATE,
        ),
    }


def evaluate_models(data, features, numeric, categorical, target, task):
    # Temporal holdout: train on forecast months Feb-Oct; test on Nov-Dec.
    forecast_month = pd.PeriodIndex(data["ForecastMonth"], freq="M")
    train_mask = forecast_month.month <= 10
    test_mask = forecast_month.month >= 11
    X_train, X_test = data.loc[train_mask, features], data.loc[test_mask, features]
    y_train, y_test = data.loc[train_mask, target], data.loc[test_mask, target]
    rows, fitted = [], {}
    for name, estimator in candidate_models().items():
        pipe = Pipeline([("prep", preprocessor(numeric, categorical)), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = np.clip(pipe.predict(X_test), 0, None)
        rows.append({
            "Task": task, "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": mean_squared_error(y_test, pred) ** 0.5,
            "R2": r2_score(y_test, pred),
            "TrainRows": len(X_train), "TestRows": len(X_test),
        })
        fitted[name] = pipe
    metrics = pd.DataFrame(rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    best_name = metrics.iloc[0]["Model"]
    return metrics, fitted[best_name], best_name, (X_test, y_test)


def importance_table(pipe, numeric, categorical, task):
    prep = pipe.named_steps["prep"]
    names = prep.get_feature_names_out()
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
    else:
        return pd.DataFrame(columns=["Task", "Feature", "Importance"])
    clean = [n.replace("num__", "").replace("cat__", "") for n in names]
    imp = pd.DataFrame({"Task": task, "Feature": clean, "Importance": values})
    return imp.sort_values("Importance", ascending=False).head(20)


def main():
    course_panel = build_course_month_dataset()
    category_panel = build_category_month_dataset(course_panel)

    demand_metrics, demand_model, demand_best, _ = evaluate_models(
        course_panel, FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
        "TargetEnrollmentCount", "Enrollment Demand"
    )
    revenue_metrics, revenue_model, revenue_best, _ = evaluate_models(
        course_panel, FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
        "TargetCourseRevenue", "Course Revenue"
    )
    category_metrics, category_model, category_best, _ = evaluate_models(
        category_panel, CATEGORY_FEATURES, CATEGORY_NUMERIC, CATEGORY_CATEGORICAL,
        "TargetCategoryRevenue", "Category Revenue"
    )

    metrics = pd.concat([demand_metrics, revenue_metrics, category_metrics], ignore_index=True)
    importance = pd.concat([
        importance_table(demand_model, NUMERIC_FEATURES, CATEGORICAL_FEATURES, "Enrollment Demand"),
        importance_table(revenue_model, NUMERIC_FEATURES, CATEGORICAL_FEATURES, "Course Revenue"),
        importance_table(category_model, CATEGORY_NUMERIC, CATEGORY_CATEGORICAL, "Category Revenue"),
    ], ignore_index=True)

    bundle = {
        "demand_model": demand_model,
        "course_revenue_model": revenue_model,
        "category_revenue_model": category_model,
        "course_features": FEATURES,
        "category_features": CATEGORY_FEATURES,
        "best_models": {
            "Enrollment Demand": demand_best,
            "Course Revenue": revenue_best,
            "Category Revenue": category_best,
        },
        "categories": sorted(course_panel["CourseCategory"].unique()),
        "levels": sorted(course_panel["CourseLevel"].unique()),
        "course_types": sorted(course_panel["CourseType"].unique()),
        "training_period": {
            "first_month": course_panel["Month"].min(),
            "last_forecast_month": course_panel["ForecastMonth"].max(),
            "holdout_months": ["2025-11", "2025-12"],
        },
    }
    joblib.dump(bundle, BASE / "model_bundle.joblib")
    metrics.to_csv(BASE / "model_metrics.csv", index=False)
    importance.to_csv(BASE / "feature_importance.csv", index=False)
    course_panel.to_csv(BASE / "course_month_forecasting.csv", index=False)
    category_panel.to_csv(BASE / "category_month_forecasting.csv", index=False)
    with open(BASE / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(bundle["best_models"] | bundle["training_period"], f, indent=2)
    print(metrics.to_string(index=False))
    print("\nBest models:", bundle["best_models"])


if __name__ == "__main__":
    main()
