# Social Protection Data System Simulation in Rwanda

This open-source simulation is inspired by my work at the Rwanda Social Security Board (RSSB). It demonstrates how synthetic data, analytics, and machine learning can be used to explore social protection coverage and inclusion in a transparent, reproducible way. In this project, I'm entirely using without using any confidential data!

---

## Overview

- Generates 100k+ rows of fully synthetic citizen-level data (age, gender, income, district, scheme type, etc.)  
- Cleans and aggregates the data via a lightweight Python ETL pipeline  
- Produces key indicators:
  - Enrollment rate by district  
  - Gender gap in coverage  
  - Income-level equity in enrollment  
- Includes a predictive ML model estimating who’s least likely to be covered  
- Outputs dynamic charts and metrics for visualization  

---

## Interactive Dashboard

The data pipeline connects to a lightweight Streamlit dashboard, which allows users to explore coverage and benefit metrics interactively

**Features**
- Filter by district and scheme type  
- View key KPIs: enrollment rate, gender gap, benefit distribution  
- Explore analytics:
  - Enrollment rate by district  
  - Predicted under-coverage risk (AI model)  
  - Gender composition among enrolled (pie chart)  
  - Enrollment by income level  
- Export cleaned data as CSV  

To run locally:
```bash
streamlit run src/dashboard_app.py
```

---

## Folder Structure

```
data/
synthetic_social_protection.csv
coverage_metrics.csv
src/
generate_data.py
cleaning_pipeline.py
dashboard_app.py
notebooks/
coverage_analysis.ipynb
figures/
*.png

```

---

## Tech Stack

`Python` · `Pandas` · `Matplotlib` · `Altair` · `scikit-learn` · `Streamlit` · `SQLite`

---

## Findings & Discussion

This simulation reproduces familiar real-world coverage patterns in social protection systems:

- Coverage disparities: Some districts show higher simulated enrollment rates, reflecting how geography and infrastructure can influence access
- Income-level inequality: Lower-income groups display lower simulated enrollment rates, reflecting structural barriers to participation
- Gender representation: Enrollment is roughly balanced but small gender gaps persist
- Benefit distribution: The benefit curve is positively skewed, with most participants receiving smaller benefits and a small minority receiving larger ones

Although synthetic, these results illustrate how data pipelines and ML can support evidence-based governance and inclusion analytics.

---

## Why This Project

Social protection data is often sensitive and fragmented. This simulation shows how safe, open, and reproducible systems can be used to model coverage, test analytics workflows, and evaluate fairness, all using synthetic, anonymized data


---

**Created by Christa Rusanganwa Ingabire**
