from pathlib import Path
import io
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="EduPro Forecasting", page_icon="📚", layout="wide")
BASE = Path(__file__).resolve().parent

st.markdown("""
<style>
/* Main application */
.stApp {
    background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
    color: #0F172A;
}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

#MainMenu, footer, header {
    visibility: hidden;
}

/* Main-page text */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] span,
[data-testid="stMarkdownContainer"] {
    color: #0F172A;
}

h1 {
    color: #1D4ED8 !important;
    font-weight: 800 !important;
}

h2, h3, h4 {
    color: #0F172A !important;
}

.stCaption, [data-testid="stCaptionContainer"] {
    color: #475569 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] div {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 9px 11px;
    border-radius: 10px;
    margin-bottom: 3px;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.10);
}

/* KPI cards */
div[data-testid="metric-container"] {
    background: #FFFFFF;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 8px 24px rgba(15,23,42,0.10);
    border-left: 6px solid #2563EB;
}

div[data-testid="metric-container"] label,
div[data-testid="metric-container"] label p {
    color: #475569 !important;
    -webkit-text-fill-color: #475569 !important;
    font-weight: 600 !important;
}

div[data-testid="metric-container"] [data-testid="stMetricValue"],
div[data-testid="metric-container"] [data-testid="stMetricValue"] div {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    font-weight: 800 !important;
}

/* Form labels */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label,
div[data-testid="stSlider"] label,
div[data-testid="stTextInput"] label {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    font-weight: 650 !important;
}

/* Number and text inputs */
div[data-testid="stNumberInput"] div[data-baseweb="input"],
div[data-testid="stTextInput"] div[data-baseweb="input"] {
    background: #FFFFFF !important;
    border: 1px solid #94A3B8 !important;
    border-radius: 12px !important;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background: #FFFFFF !important;
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    caret-color: #0F172A !important;
    font-weight: 600 !important;
}

div[data-testid="stNumberInput"] button {
    background: #1E293B !important;
    color: #FFFFFF !important;
}

div[data-testid="stNumberInput"] button svg {
    fill: #FFFFFF !important;
}

/* Selectbox and multiselect */
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border: 1px solid #94A3B8 !important;
    border-radius: 12px !important;
    color: #0F172A !important;
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] span,
div[data-testid="stSelectbox"] div[data-baseweb="select"] div,
div[data-testid="stMultiSelect"] div[data-baseweb="select"] span {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}

div[data-testid="stSelectbox"] svg,
div[data-testid="stMultiSelect"] svg {
    fill: #334155 !important;
}

/* Selected multiselect tags */
div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: #DBEAFE !important;
    color: #1E3A8A !important;
}

div[data-testid="stMultiSelect"] span[data-baseweb="tag"] span,
div[data-testid="stMultiSelect"] span[data-baseweb="tag"] div {
    color: #1E3A8A !important;
    -webkit-text-fill-color: #1E3A8A !important;
}

/* Open dropdown menu */
div[data-baseweb="popover"],
div[data-baseweb="popover"] ul {
    background: #FFFFFF !important;
}

div[data-baseweb="popover"] li,
div[data-baseweb="popover"] li span,
div[role="option"],
div[role="option"] span {
    background: #FFFFFF !important;
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}

div[data-baseweb="popover"] li:hover,
div[role="option"]:hover {
    background: #DBEAFE !important;
    color: #1D4ED8 !important;
}

/* Sliders */
div[data-testid="stSlider"] [data-testid="stTickBar"] div,
div[data-testid="stSlider"] [role="slider"] {
    color: #0F172A !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button,
div[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    min-height: 50px;
    border: none;
    border-radius: 12px;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 700;
}

.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(90deg, #2563EB, #4F46E5);
}

.stDownloadButton > button {
    background: #059669;
}

.stButton > button p,
.stDownloadButton > button p,
div[data-testid="stFormSubmitButton"] > button p {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 14px;
}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] li,
div[data-testid="stAlert"] div {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}

/* Charts and tables */
div[data-testid="stPlotlyChart"],
div[data-testid="stDataFrame"] {
    background: #FFFFFF;
    padding: 10px;
    border-radius: 16px;
    box-shadow: 0 6px 20px rgba(15,23,42,0.08);
}

/* Help tooltips */
div[data-testid="stTooltipIcon"] svg {
    fill: #334155 !important;
}

@media only screen and (max-width: 768px) {
    .block-container {padding-left: 1rem; padding-right: 1rem;}
    h1 {font-size: 30px !important;}
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv(BASE / "Merged_EduPro.csv")
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    course_panel = pd.read_csv(BASE / "course_month_forecasting.csv")
    category_panel = pd.read_csv(BASE / "category_month_forecasting.csv")
    metrics = pd.read_csv(BASE / "model_metrics.csv")
    importance = pd.read_csv(BASE / "feature_importance.csv")
    return df, course_panel, category_panel, metrics, importance


@st.cache_resource
def load_bundle():
    return joblib.load(BASE / "model_bundle.joblib")


try:
    df, course_panel, category_panel, metrics, importance = load_data()
    bundle = load_bundle()
except Exception as exc:
    st.error(f"Project files could not be loaded: {exc}")
    st.stop()


def price_band(price):
    if price <= 0:
        return "Free"
    return "Low" if price <= 250 else "High"


def duration_bucket(duration):
    if duration <= 20:
        return "Short"
    return "Medium" if duration <= 40 else "Long"


def rating_tier(rating):
    if rating <= 3:
        return "Developing"
    return "Good" if rating <= 4 else "Excellent"


def experience_bucket(years):
    if years <= 5:
        return "Early Career"
    return "Experienced" if years <= 12 else "Senior"


def plot_theme(fig, height=430):
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=20,r=20,t=55,b=20),
        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        font=dict(color="white"), title_font=dict(color="white"),
    )
    return fig


def demand_label(value):
    q1, q2 = course_panel["TargetEnrollmentCount"].quantile([0.33, 0.67])
    if value < q1:
        return "Low Demand"
    if value < q2:
        return "Moderate Demand"
    return "High Demand"


st.sidebar.markdown("<div style='text-align:center'><div style='font-size:52px'>📚</div><h2>EduPro</h2><p>Predictive Intelligence</p></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "🏠 Home", "📊 Historical Insights", "🔮 Course Forecast",
    "🏷️ Category Forecast", "📈 Model Performance", "ℹ️ About",
])
st.sidebar.markdown("---")
st.sidebar.success("**Forecasting Approach**\n\nMonth-ahead prediction\n\nTime-based holdout")
st.sidebar.info("**Developer**\n\nAman Jat\n\n**Internship**\n\nUnified Mentor")


if page == "🏠 Home":
    st.markdown("<h1 style='text-align:center;font-size:50px'>📚 EduPro Predictive Intelligence Dashboard</h1><p style='text-align:center;font-size:20px;color:#64748B'>Course demand, course revenue, and category revenue forecasting</p>", unsafe_allow_html=True)
    st.markdown("---")
    a,b,c,d = st.columns(4)
    a.metric("👥 Users", f"{df.UserID.nunique():,}")
    b.metric("📚 Courses", f"{df.CourseID.nunique():,}")
    c.metric("🧑‍🏫 Teachers", f"{df.TeacherID.nunique():,}")
    d.metric("💰 Historical Revenue", f"₹{df.Amount.sum():,.2f}")
    st.markdown("## 🎯 Business Problem")
    st.info("""
EduPro needs forward-looking evidence for **course launches, pricing, instructor planning, and resource allocation**. This application converts historical course and transaction data into three month-ahead targets:

- expected course enrollment count;
- expected revenue per course; and
- expected aggregated revenue by course category.
""")
    st.markdown("## ✅ Completed Modules")
    c1,c2,c3,c4 = st.columns(4)
    c1.success("Course demand prediction")
    c2.success("Course revenue forecast")
    c3.success("Category revenue forecast")
    c4.success("Feature importance explorer")
    st.markdown("## 🧭 Workflow")
    st.write("Data preparation → monthly course aggregation → engineered historical features → time-based validation → model comparison → interactive forecasting")


elif page == "📊 Historical Insights":
    st.title("📊 Historical Insights")
    st.caption("Explore revenue, enrollment, learner, and course-performance patterns before forecasting.")
    fc1,fc2 = st.columns(2)
    categories = sorted(df.CourseCategory.dropna().unique())
    payments = sorted(df.PaymentMethod.dropna().unique())
    chosen_categories = fc1.multiselect("Course categories", categories, default=categories)
    chosen_payments = fc2.multiselect("Payment methods", payments, default=payments)
    f = df[df.CourseCategory.isin(chosen_categories) & df.PaymentMethod.isin(chosen_payments)].copy()
    if f.empty:
        st.warning("Select at least one category and payment method.")
        st.stop()
    a,b,c,d = st.columns(4)
    a.metric("Revenue", f"₹{f.Amount.sum():,.2f}")
    b.metric("Transactions", f"{f.TransactionID.nunique():,}")
    c.metric("Courses", f"{f.CourseID.nunique():,}")
    d.metric("Users", f"{f.UserID.nunique():,}")
    f["Month"] = f.TransactionDate.dt.to_period("M").astype(str)
    monthly = f.groupby("Month", as_index=False).Amount.sum()
    st.plotly_chart(plot_theme(px.line(monthly,x="Month",y="Amount",markers=True,title="Monthly Revenue Trend")), use_container_width=True)
    left,right = st.columns(2)
    cat = f.groupby("CourseCategory",as_index=False).Amount.sum().sort_values("Amount")
    left.plotly_chart(plot_theme(px.bar(cat,x="Amount",y="CourseCategory",orientation="h",title="Revenue by Category")),use_container_width=True)
    pay = f.groupby("PaymentMethod",as_index=False).Amount.sum()
    right.plotly_chart(plot_theme(px.pie(pay,names="PaymentMethod",values="Amount",hole=.5,title="Revenue by Payment Method")),use_container_width=True)
    demand = f.groupby("CourseName",as_index=False).TransactionID.count().nlargest(10,"TransactionID").sort_values("TransactionID")
    revenue = f.groupby("CourseName",as_index=False).Amount.sum().nlargest(10,"Amount").sort_values("Amount")
    left,right = st.columns(2)
    left.plotly_chart(plot_theme(px.bar(demand,x="TransactionID",y="CourseName",orientation="h",title="Top Courses by Enrollment")),use_container_width=True)
    right.plotly_chart(plot_theme(px.bar(revenue,x="Amount",y="CourseName",orientation="h",title="Top Courses by Revenue")),use_container_width=True)


elif page == "🔮 Course Forecast":
    st.title("🔮 Course Demand & Revenue Forecast")
    st.caption("Enter course, instructor, and recent performance information to forecast the next month.")
    with st.form("course_forecast"):
        c1,c2,c3 = st.columns(3)
        price = c1.number_input("Course price (₹)",0.0,1000.0,250.0,10.0)
        duration = c1.number_input("Course duration (hours)",1.0,100.0,30.0,1.0)
        course_rating = c1.slider("Course rating",1.0,5.0,4.0,0.1)
        course_type = c2.selectbox("Course type",bundle["course_types"])
        level = c2.selectbox("Course level",bundle["levels"])
        category = c2.selectbox("Course category",bundle["categories"])
        experience = c3.number_input("Instructor experience (years)",0,40,8)
        teacher_rating = c3.slider("Instructor rating",1.0,5.0,4.0,0.1)
        expertise_match = c3.slider("Expertise-category match",0.0,1.0,0.7,0.05,help="1.0 means the instructor expertise fully matches the course category.")
        st.markdown("### Recent historical performance")
        h1,h2,h3,h4 = st.columns(4)
        past_enrollment = h1.number_input("Past-month enrollments",0,1000,15)
        past_avg_revenue = h2.number_input("Historical average revenue (₹)",0.0,100000.0,2000.0,100.0)
        rev_per_enrollment = h3.number_input("Revenue per enrollment (₹)",0.0,1000.0,float(price),10.0)
        forecast_month = h4.selectbox("Forecast month",list(range(1,13)),index=11,format_func=lambda x:pd.Timestamp(2025,x,1).strftime("%B"))
        submitted = st.form_submit_button("Generate Forecast")
    if submitted:
        row = pd.DataFrame([{
            "CoursePrice":price,"CourseDuration":duration,"CourseRating":course_rating,
            "YearsOfExperience":experience,"TeacherRating":teacher_rating,
            "ExpertiseCategoryMatch":expertise_match,"PastEnrollmentCount":past_enrollment,
            "PastAverageRevenue":past_avg_revenue,"RevenuePerEnrollment":rev_per_enrollment,
            "MonthNumber":forecast_month,"CourseType":course_type,"CourseLevel":level,
            "CourseCategory":category,"PriceBand":price_band(price),
            "DurationBucket":duration_bucket(duration),"RatingTier":rating_tier(course_rating),
            "ExperienceBucket":experience_bucket(experience),
        }])
        enrollment = max(0,float(bundle["demand_model"].predict(row[bundle["course_features"]])[0]))
        revenue = max(0,float(bundle["course_revenue_model"].predict(row[bundle["course_features"]])[0]))
        m1,m2,m3 = st.columns(3)
        m1.metric("Expected Enrollments",f"{enrollment:.0f}")
        m2.metric("Demand Category",demand_label(enrollment))
        m3.metric("Expected Course Revenue",f"₹{revenue:,.2f}")
        st.success("The forecast uses only information available before the forecast month.")
        summary = row.copy()
        summary["PredictedEnrollments"] = round(enrollment,2)
        summary["PredictedCourseRevenue"] = round(revenue,2)
        summary["DemandCategory"] = demand_label(enrollment)
        st.dataframe(summary.T.rename(columns={0:"Value"}),use_container_width=True)
        st.download_button("Download Forecast CSV",summary.to_csv(index=False).encode(),"EduPro_course_forecast.csv","text/csv")


elif page == "🏷️ Category Forecast":
    st.title("🏷️ Category-Level Revenue Forecast")
    st.caption("Compare predicted next-month revenue across all course categories.")
    latest = category_panel.sort_values("Month").groupby("CourseCategory",as_index=False).tail(1).copy()
    chosen_month = st.selectbox("Forecast month",list(range(1,13)),index=11,format_func=lambda x:pd.Timestamp(2025,x,1).strftime("%B"))
    latest["MonthNumber"] = chosen_month
    latest["PredictedCategoryRevenue"] = np.clip(
        bundle["category_revenue_model"].predict(latest[bundle["category_features"]]),0,None
    )
    ranked = latest.sort_values("PredictedCategoryRevenue",ascending=False)
    fig = px.bar(ranked,x="CourseCategory",y="PredictedCategoryRevenue",color="PredictedCategoryRevenue",title="Forecast Revenue by Category",labels={"PredictedCategoryRevenue":"Predicted revenue (₹)"})
    st.plotly_chart(plot_theme(fig,500),use_container_width=True)
    a,b,c = st.columns(3)
    a.metric("Top Forecast Category",ranked.iloc[0].CourseCategory)
    b.metric("Top Forecast Revenue",f"₹{ranked.iloc[0].PredictedCategoryRevenue:,.2f}")
    c.metric("Total Category Forecast",f"₹{ranked.PredictedCategoryRevenue.sum():,.2f}")
    st.dataframe(ranked[["CourseCategory","PastEnrollmentCount","PastAverageRevenue","PredictedCategoryRevenue"]],use_container_width=True,hide_index=True)
    st.download_button("Download Category Forecast",ranked.to_csv(index=False).encode(),"EduPro_category_forecast.csv","text/csv")


elif page == "📈 Model Performance":
    st.title("📈 Model Performance & Feature Importance")
    st.info("Models are evaluated using a **time-based holdout**: forecast months November and December are kept out of training.")
    for task in ["Enrollment Demand","Course Revenue","Category Revenue"]:
        part = metrics[metrics.Task==task].sort_values("RMSE")
        best = part.iloc[0]
        st.markdown(f"## {task}")
        a,b,c,d = st.columns(4)
        a.metric("Best Model",best.Model)
        b.metric("MAE",f"{best.MAE:,.3f}")
        c.metric("RMSE",f"{best.RMSE:,.3f}")
        d.metric("R²",f"{best.R2:,.3f}")
        st.dataframe(part[["Model","MAE","RMSE","R2","TrainRows","TestRows"]],use_container_width=True,hide_index=True)
    selected_task = st.selectbox("Feature-importance task",importance.Task.unique())
    imp = importance[importance.Task==selected_task].nlargest(15,"Importance").sort_values("Importance")
    st.plotly_chart(plot_theme(px.bar(imp,x="Importance",y="Feature",orientation="h",title=f"Top Drivers: {selected_task}"),520),use_container_width=True)
    st.warning("Demand R² is low because month-to-month enrollments in the supplied synthetic data are weakly predictable. The dashboard reports this honestly instead of using the earlier leakage-prone transaction split.")


else:
    st.title("ℹ️ About the Project")
    st.info("**Project Title:** Predictive Modeling for Course Demand and Revenue Forecasting on EduPro")
    st.markdown("## Problem Statement")
    st.write("EduPro lacked predictive models for enrollment demand, course- and category-level revenue forecasting, and quantitative evidence for launch and pricing decisions. This project introduces forward-looking intelligence for proactive planning.")
    st.markdown("## Predictive Targets")
    st.markdown("- Enrollment count per course\n- Total revenue generated per course\n- Aggregated revenue by course category")
    st.markdown("## Required Feature Engineering")
    st.markdown("- Price bands, duration buckets, rating tiers, and course-level encoding\n- Instructor-experience buckets, teacher rating, and expertise-category match\n- Past enrollment count, historical average revenue, and revenue per enrollment")
    st.markdown("## Methodology")
    st.markdown("1. Merge and aggregate the source data at course-month level.\n2. Build only past-information features.\n3. Compare Linear, Ridge, Lasso, Random Forest, and Gradient Boosting models.\n4. Evaluate MAE, RMSE, and R² on future months.\n5. Translate model outputs into planning, pricing, and allocation insights.")
    st.markdown("## Conclusion")
    st.success("EduPro can now use historical data to estimate future demand and revenue, compare categories, support course-roadmap decisions, optimize pricing, and allocate resources more effectively.")
    st.markdown("## Technologies")
    st.write("Python • pandas • NumPy • scikit-learn • Plotly • Streamlit • joblib")
    