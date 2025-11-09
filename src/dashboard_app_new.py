import os
import sqlite3
import pandas as pd
import streamlit as st
import altair as alt

# ----------------- Page config -----------------
st.set_page_config(page_title="Rwanda Social Protection Dashboard", page_icon="🌍", layout="wide")

# ---------- Data loaders ----------
@st.cache_data(show_spinner=False)
def load_metrics():
    db_path = "data/social_protection.db"
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM coverage_metrics", conn)
        conn.close()
    else:
        df = pd.read_csv("data/coverage_metrics.csv")
    return df

@st.cache_data(show_spinner=False)
def load_people():
    return pd.read_csv("data/synthetic_social_protection.csv")

metrics = load_metrics()
people = load_people()

# ---------- Styling ----------
st.markdown(
    """
    <style>
        .big-title {font-size:2.1rem; font-weight:700; text-align:center; margin-bottom:0.2rem;}
        .subhead {text-align:center; font-size:1.1rem; color:#555; margin-bottom:1.5rem;}
    </style>
    """, unsafe_allow_html=True)

st.markdown("<p class='big-title'> Rwanda Social Protection Dashboard</p>", unsafe_allow_html=True)
st.markdown("<p class='subhead'>Synthetic simulation of coverage and benefits (inspired by real-world systems at RSSB)</p>", unsafe_allow_html=True)

# ---------- Filters ----------
st.sidebar.header("Filters")
sel_districts = st.sidebar.multiselect("District(s)", sorted(metrics["district"].dropna().unique()))
sel_schemes = st.sidebar.multiselect("Scheme(s)", sorted(people["scheme"].dropna().unique()))

filtered_people = people.copy()
if sel_districts:
    filtered_people = filtered_people[filtered_people["district"].isin(sel_districts)]
if sel_schemes:
    filtered_people = filtered_people[filtered_people["scheme"].isin(sel_schemes)]

# ---------- Helpers ----------
def pct(x): 
    return "—" if pd.isna(x) else f"{x*100:.1f}%"
def money(x): 
    return "—" if pd.isna(x) else f"{x:,.0f} RWF"

# ---------- KPIs ----------
overall_enroll = filtered_people["enrolled"].mean()
cov_by_gender = filtered_people.groupby("gender")["enrolled"].mean()
cov_f = cov_by_gender.get("Female", float("nan"))
cov_m = cov_by_gender.get("Male", float("nan"))
gender_gap = cov_m - cov_f if pd.notna(cov_m) and pd.notna(cov_f) else float("nan")

avg_benefit = filtered_people.loc[filtered_people["enrolled"], "benefit_amount"].mean()
n_enrolled = int(filtered_people["enrolled"].sum())
dist_enroll = filtered_people.groupby("district")["enrolled"].mean().sort_values(ascending=False)
top_district = dist_enroll.index[0] if not dist_enroll.empty else "—"
top_rate = dist_enroll.iloc[0] if not dist_enroll.empty else float("nan")

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Enrollment Rate", pct(overall_enroll))
k2.metric("Gender Gap (M−F)", pct(gender_gap))
k3.metric("Average Benefit (All Enrolled)", money(avg_benefit))
k4.metric("Enrolled People", f"{n_enrolled:,}")
k5.metric("Top District", f"{top_district}: {pct(top_rate)}")

st.divider()

# ---------- Charts ----------
col1, col2 = st.columns(2)

# Enrollment by district
with col1:
    st.subheader("Enrollment Rate by District")
    chart = (
        alt.Chart(filtered_people.groupby("district", observed=True)["enrolled"]
                  .mean().reset_index())
        .mark_bar(color="#005C8F")
        .encode(
            x=alt.X("district:N", sort="-y", title="District"),
            y=alt.Y("enrolled:Q", title="Enrollment Rate", axis=alt.Axis(format=".0%")),
            tooltip=[alt.Tooltip("district:N"), alt.Tooltip("enrolled:Q", format=".1%")]
        )
        .properties(height=380)
    )
    st.altair_chart(chart, use_container_width=True)

# ---------- GENDER CHART (choose one) ----------
with col2:
    st.subheader("Gender Composition Among Enrolled")

    # compute gender share
    gender_share = (
        filtered_people[filtered_people["enrolled"] == 1]
        .groupby("gender")["person_id"].count()
        .reset_index(name="count")
    )
    if not gender_share.empty:
        gender_share["share"] = gender_share["count"] / gender_share["count"].sum()

        # 1 PIE CHART VERSION
        pie = (
            alt.Chart(gender_share)
            .mark_arc(innerRadius=70)
            .encode(
                theta=alt.Theta(field="share", type="quantitative"),
                color=alt.Color(field="gender", type="nominal", scale=alt.Scale(scheme="pastel2")),
                tooltip=[alt.Tooltip("gender:N"), alt.Tooltip("share:Q", format=".1%")]
            )
            .properties(height=380)
        )
        st.altair_chart(pie, use_container_width=True)

        # 2 (optional alternative) gauge style
        # if you prefer, replace above with:
        # st.metric("Female Share", pct(gender_share.loc[gender_share["gender"]=="Female","share"].values[0]))
    else:
        st.info("No enrolled records for selected filters.")

st.divider()

# Enrollment by income (equity)
c1, c2 = st.columns(2)
with c1:
    st.subheader("Enrollment by Income Level (Equity)")
    income_df = (
        filtered_people.groupby("income_level", observed=True)["enrolled"]
        .mean().rename("enrollment_rate").reset_index()
    )
    bar = (
        alt.Chart(income_df)
        .mark_bar(color="#1F77B4")
        .encode(
            x=alt.X("income_level:N", title="Income Level", sort=["Low","Lower-Middle","Upper-Middle","High"]),
            y=alt.Y("enrollment_rate:Q", axis=alt.Axis(format=".0%"), title="Enrollment Rate")
        )
        .properties(height=380)
    )
    st.altair_chart(bar, use_container_width=True)

with c2:
    st.subheader("Benefit Distribution (Enrolled)")
    enr = filtered_people[filtered_people["enrolled"] == 1]
    if not enr.empty:
        hist = alt.Chart(enr).mark_bar(color="#2CA02C").encode(
            x=alt.X("benefit_amount:Q", bin=alt.Bin(maxbins=30), title="Benefit (RWF)"),
            y=alt.Y("count()", title="Count")
        ).properties(height=380)
        st.altair_chart(hist, use_container_width=True)
    else:
        st.info("No enrolled records for selected filters.")

st.divider()

st.markdown("### Export Options")
colx1, colx2 = st.columns(2)
with colx1:
    st.download_button(
        "Download Coverage Metrics (CSV)",
        data=metrics.to_csv(index=False),
        file_name="coverage_metrics.csv",
        mime="text/csv"
    )
with colx2:
    st.download_button(
        "Download People Dataset (CSV)",
        data=people.to_csv(index=False),
        file_name="synthetic_social_protection.csv",
        mime="text/csv"
    )

st.caption("Data source: Synthetic dataset generated with Python (no confidential data).  © Christa Ingabire, 2025")
