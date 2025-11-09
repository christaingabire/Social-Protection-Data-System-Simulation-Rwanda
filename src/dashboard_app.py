import os
import sqlite3
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# ----------------- Page config -----------------
st.set_page_config(page_title="Rwanda Social Protection Dashboard", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")


# ---------- Data loaders ----------
@st.cache_data(show_spinner=False)
def load_metrics():
    db_path = "data/social_protection.db"
    if os.path.exists(db_path):
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql("SELECT * FROM coverage_metrics", conn)
    else:
        df = pd.read_csv("data/coverage_metrics.csv")
    return df


@st.cache_data(show_spinner=False)
def load_people():
    return pd.read_csv("data/synthetic_social_protection.csv")


metrics = load_metrics()
people = load_people()


# ---------- Helper functions ----------
def pct(x):
    if pd.isna(x):
        return "—"
    return f"{x * 100:.1f}%"

def money(x):
    if pd.isna(x):
        return "—"
    return f"{round(x):,} RWF"

def fmt_int(x):
    return f"{int(x):,}" if not pd.isna(x) else "—"


# ---------- Model trainer ----------
@st.cache_data(show_spinner=False)
def train_coverage_model(df_people: pd.DataFrame):
    feats = ["age", "gender", "district", "education_level", "income_level", "scheme"]
    X = df_people[feats].copy()
    y = df_people["enrolled"].astype(int)
    if y.nunique() < 2:
        return None

    cat_cols = ["gender", "district", "education_level", "income_level", "scheme"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", ["age"]),
    ])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=600))])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model.fit(Xtr, ytr)
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])

    ohe = model.named_steps["pre"].named_transformers_["cat"]
    names = list(ohe.get_feature_names_out(cat_cols)) + ["age"]
    coefs = model.named_steps["clf"].coef_.ravel()
    coef_df = pd.DataFrame({"feature": names, "coef": coefs}).sort_values("coef", ascending=False)
    return {"auc": auc, "coef_df": coef_df}


# ---------- Page styling ----------
st.markdown(
    """
    <style>
        body {background-color: #fcfcfc;}, .stMarkdown {font-family: 'Inter', sans-serif;}
        .page-container {max-width: 1180px; margin: auto;}
        .header-block {text-align: center; padding-top: 2.8rem; padding-bottom: 1.2rem;}
        .big-title {
            font-size: 4.2rem !important;
            font-weight: 900 !important;
            color: #002b36 !important;
            letter-spacing: -0.8px !important;
            margin-top: 0.6rem !important;
            margin-bottom: 0.8rem !important;
        }
        .subhead {
            font-size: 1.35rem;
            color: #485056;
            font-weight: 400;
            line-height: 1.45;
            margin-bottom: 1.8rem;
        }
        .intro-box {
            background-color: #f8f9fa;
            border-left: 6px solid #005C8F;
            border-radius: 0.5rem;
            padding: 1rem 1.3rem;
            max-width: 850px;
            margin: 0 auto 2.3rem auto;
            color: #4b4f52;
            font-size: 1rem;
            line-height: 1.65;
        }
        .section-title {
            font-size: 1.6rem;
            font-weight: 700;
            color: #002b36;
            margin-top: 2.3rem;
            margin-bottom: 0.6rem;
        }
        .caption-note {
            color: #666;
            font-size: 0.9rem;
            margin-top: -0.4rem;
            margin-bottom: 1.4rem;
            line-height: 1.4;
        }
        div[data-testid="stMetricValue"] {
            color: #002b36;
            font-size: 1.85rem !important;
            font-weight: 750 !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.95rem !important;
            color: #555 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Header & Intro ----------
st.markdown("<div class='page-container'><div class='header-block'>", unsafe_allow_html=True)
st.markdown("<p class='big-title'>Rwanda Social Protection Dashboard</p>", unsafe_allow_html=True)
st.markdown(
    "<p class='subhead'>A simulation of national social protection coverage and benefits (inspired by systems at the Rwanda Social Security Board (RSSB))</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='intro-box'>This dashboard uses synthetic citizen-level data to illustrate how analytics and AI can support inclusive, equitable social protection design in Rwanda. It demonstrates data-driven coverage analysis without exposing confidential records.</div>",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# ---------- Filters ----------
st.sidebar.markdown("### Filters")
sel_districts = st.sidebar.multiselect("District(s)", sorted(metrics["district"].dropna().unique()))
sel_schemes = st.sidebar.multiselect("Scheme(s)", sorted(people["scheme"].dropna().unique()))

filtered_people = people.copy()
if sel_districts:
    filtered_people = filtered_people[filtered_people["district"].isin(sel_districts)]
if sel_schemes:
    filtered_people = filtered_people[filtered_people["scheme"].isin(sel_schemes)]


# ---------- KPI area ----------
#st.markdown("---")
#st.markdown("<br>", unsafe_allow_html=True)

overall_enroll = filtered_people["enrolled"].mean()
cov_by_gender = filtered_people.groupby("gender")["enrolled"].mean()
cov_f = cov_by_gender.get("Female", np.nan)
cov_m = cov_by_gender.get("Male", np.nan)
gender_gap = cov_m - cov_f if pd.notna(cov_m) and pd.notna(cov_f) else np.nan
avg_benefit = filtered_people.loc[filtered_people["enrolled"], "benefit_amount"].mean()
n_enrolled = int(filtered_people["enrolled"].sum())

k1, k2, k3, k4 = st.columns([1,1,1,1])
k1.metric("Enrollment Rate", f"{overall_enroll*100:.1f}%")
k2.metric("Gender Gap (M−F)", f"{gender_gap*100:.1f}%" if not np.isnan(gender_gap) else "—")
k3.metric("Average Benefit", f"{round(avg_benefit):,} RWF")
k4.metric("Number of Beneficiaries", f"{n_enrolled:,}")

st.markdown("---")


# ---------- Coverage and Predicted Gaps ----------
st.subheader("Coverage Patterns and Predicted Gaps")

# Enrollment Rate chart
st.markdown("<p class='caption-note'>Observed enrollment rates vary by district, reflecting access and income differences.</p>", unsafe_allow_html=True)
enroll_by_dist_df = (
    filtered_people.groupby("district", observed=True)["enrolled"]
    .mean()
    .reset_index()
    .rename(columns={"enrolled": "enrollment_rate"})
    .sort_values("enrollment_rate", ascending=False)
)
enroll_by_dist_df["enrollment_rate"] = enroll_by_dist_df["enrollment_rate"].round(3)

if not enroll_by_dist_df.empty:
    chart_enroll = (
        alt.Chart(enroll_by_dist_df)
        .mark_bar(color="#005C8F")
        .encode(
            x=alt.X("district:N", sort="-y", title="District"),
            y=alt.Y("enrollment_rate:Q",
                    title="Enrollment Rate",
                    axis=alt.Axis(format=".0%")),
            tooltip=[
                alt.Tooltip("district:N", title="District"),
                alt.Tooltip("enrollment_rate:Q", title="Enrollment Rate", format=".1%")
            ],
        )
        .properties(height=400)
    )
    st.altair_chart(chart_enroll, use_container_width=True)
else:
    st.info("No enrollment data available for selected filters.")


# Predicted Under-Coverage Risk chart
st.subheader("Predicted Coverage Gaps (AI Model)")

if "avg_undercoverage_risk" in metrics.columns:
    risk_df = metrics[["district", "avg_undercoverage_risk"]].dropna()
    risk_df["avg_undercoverage_risk"] = risk_df["avg_undercoverage_risk"].round(3)
    chart_risk = (
        alt.Chart(risk_df)
        .mark_bar(color="#E76F51")
        .encode(
            x=alt.X("district:N", sort="-y", title="District"),
            y=alt.Y("avg_undercoverage_risk:Q",
                    title="Predicted Risk",
                    axis=alt.Axis(format=".0%")),
            tooltip=[
                alt.Tooltip("district:N", title="District"),
                alt.Tooltip("avg_undercoverage_risk:Q", title="Predicted Risk", format=".1%")
            ],
        )
        .properties(height=400)
    )
    st.altair_chart(chart_risk, use_container_width=True)
    st.markdown(
    "<p class='caption-note'>Districts with taller bars represent a higher modeled likelihood of under-coverage — indicating areas where residents are statistically less likely to be enrolled.</p>",
    unsafe_allow_html=True
)

else:
    st.info("Run the pipeline first to generate predictive risk data.")


# ---------- AI Model Insights ----------
st.markdown("---")
st.subheader("Model Evaluation and Feature Importance")

st.markdown(
    "<p style='color:#444; font-size:0.95rem;'>"
    "This logistic regression model estimates the probability of enrollment using demographic and socioeconomic factors. "
    "It highlights which variables most influence coverage disparities across the population."
    "</p>", unsafe_allow_html=True)

model_pack = train_coverage_model(filtered_people)
if model_pack is None:
    st.info("Not enough variation to evaluate the model under current filters.")
else:
    auc = model_pack["auc"]
    coef_df = model_pack["coef_df"]
    st.write(f"**ROC-AUC:** {auc:.3f}  ·  AUC measures model accuracy (1.0 = perfect, 0.5 = random).")

    top_pos = coef_df.head(8)
    top_neg = coef_df.tail(8).sort_values("coef")

    cpos, cneg = st.columns(2)
    with cpos:
        st.caption("Features linked with higher enrollment probability")
        pos_chart = (
            alt.Chart(top_pos)
            .mark_bar(color="#2A9D8F")
            .encode(x="coef:Q", y=alt.Y("feature:N", sort="-x"))
            .properties(height=220)
        )
        st.altair_chart(pos_chart, use_container_width=True)

    with cneg:
        st.caption("Features linked with lower enrollment probability")
        neg_chart = (
            alt.Chart(top_neg)
            .mark_bar(color="#B23A48")
            .encode(x="coef:Q", y=alt.Y("feature:N", sort="x"))
            .properties(height=220)
        )
        st.altair_chart(neg_chart, use_container_width=True)


# ---------- Gender and Income ----------
st.markdown("---")

# Gender chart
st.subheader("Gender Composition Among Enrolled")

gender_share = (
    filtered_people[filtered_people["enrolled"] == 1]
    .groupby("gender")["person_id"].count().reset_index(name="count")
)
if not gender_share.empty:
    gender_share["share"] = gender_share["count"] / gender_share["count"].sum()
    pie = (
        alt.Chart(gender_share)
        .mark_arc(innerRadius=70)
        .encode(
            theta=alt.Theta("share:Q"),
            color=alt.Color("gender:N", scale=alt.Scale(scheme="pastel2")),
            tooltip=[
                alt.Tooltip("gender:N", title="Gender"),
                alt.Tooltip("share:Q", title="Share", format=".1%")
            ],
        )
        .properties(height=380)
    )
    st.altair_chart(pie, use_container_width=True)
else:
    st.info("No enrolled records for selected filters.")

st.divider()

# Income-level bar chart
st.subheader("Enrollment by Income Level")

income_df = (
    filtered_people.groupby("income_level", observed=True)["enrolled"]
    .mean().rename("enrollment_rate").reset_index()
)
if not income_df.empty:
    income_df["enrollment_rate"] = income_df["enrollment_rate"].round(3)
    income_chart = (
        alt.Chart(income_df)
        .mark_bar(color="#1F77B4")
        .encode(
            x=alt.X("income_level:N",
                    sort=["Low", "Lower-Middle", "Upper-Middle", "High"],
                    title="Income Level"),
            y=alt.Y("enrollment_rate:Q",
                    axis=alt.Axis(format=".0%", title="Enrollment Rate")),
            tooltip=[
                alt.Tooltip("income_level:N", title="Income Level"),
                alt.Tooltip("enrollment_rate:Q", title="Rate", format=".1%")
            ]
        )
        .properties(height=380)
    )
    st.altair_chart(income_chart, use_container_width=True)
else:
    st.info("No enrollment data available for selected filters.")


# ---------- Benefit Distribution ----------

st.divider()
st.subheader("Benefit Distribution Overview")

enrolled = filtered_people[filtered_people["enrolled"] == 1]
if not enrolled.empty:
    hist = (
        alt.Chart(enrolled)
        .mark_bar(color="#264653")
        .encode(
            x=alt.X("benefit_amount:Q", bin=alt.Bin(maxbins=30), title="Benefit (RWF)"),
            y=alt.Y("count()", title="Number of Beneficiaries"),
        )
        .properties(height=360)
    )
    st.altair_chart(hist, use_container_width=True)
else:
    st.info("No enrolled records for selected filters.")

st.markdown("---")

# ---------- Export ----------

st.markdown("<p class='section-title'>Export Options</p>", unsafe_allow_html=True)
colx1, colx2 = st.columns(2)
with colx1:
    st.download_button("Download Coverage Metrics (CSV)",
        data=metrics.to_csv(index=False),
        file_name="coverage_metrics.csv", mime="text/csv")
with colx2:
    st.download_button("Download People Dataset (CSV)",
        data=people.to_csv(index=False),
        file_name="synthetic_social_protection.csv", mime="text/csv")

st.caption("Data source: synthetic dataset generated with Python (no confidential data). © Christa Ingabire, 2025")
