# Social Protection Data System Simulation — Rwanda

This project is a small, open-source simulation inspired by my work at the Rwanda Social Security Board. It models how data pipelines and analytics can help governments monitor coverage, identify inequalities, and design more inclusive social protection systems. In this project I'm entirely using **entirely synthetic data!**

---

### Overview

- Generates 100k+ rows of synthetic citizen-level data (age, gender, income, district, scheme type, etc.)
- Cleans and aggregates the data using a lightweight Python ETL pipeline
- Produces key indicators:
  - Enrollment rate by district  
  - Gender gap in coverage  
  - Urban–rural benefit disparities  
- Includes a small predictive model (logistic regression) to estimate who is least likely to be covered
- All visuals are generated automatically through the pipeline

---

### Folder Structure

```
data/
synthetic_social_protection.csv
coverage_metrics.csv
src/
generate_data.py
cleaning_pipeline.py
notebooks/
coverage_analysis.ipynb
figures/
*.png

```

---

### Tech Stack

Python · Pandas · Matplotlib · Seaborn · scikit-learn · Jupyter Notebook

---

### Why I Built It

Social protection data is often sensitive and fragmented. I wanted to show, in a transparent and ethical way, how open data systems could be designed to make coverage metrics visible and actionable. This small simulation captures that idea in a controlled, reproducible way

---

### Next Steps

- Add a database layer (SQLite or DuckDB) for versioned metrics  
- Explore an interactive dashboard (Streamlit or Superset)  
- Document lessons for **AI for Social Good** and **Data for Governance** work

---

### Findings & Discussion

This simulation highlights structural patterns that often emerge in real social protection systems (even within synthetic data)

- **Coverage disparities:** Enrollment is uneven across districts, reflecting how geography and infrastructure can shape access to formal social protection. Some districts reach simulated enrollment rates above 55%, while others remain below 45%  
- **Income-level inequality:** Lower-income groups have visibly lower simulated enrollment rates, while higher-income individuals are consistently more represented in social protection schemes. This mirrors how affordability, formality of employment, and information gaps influence participation  
- **Gender representation:** The gender composition among enrolled individuals is roughly balanced in this simulation, but small gender gaps persist. This suggests that gender-focused outreach, especially for community-based and informal-sector schemes, remains relevant  
- **Benefit distribution:** The benefit curve is positively skewed, meaning most participants receive smaller benefits while a small subset receives higher payouts. This reflects typical inequality in contributory systems, where benefit amounts depend on income and scheme type

Although entirely synthetic, these patterns illustrate how data pipelines and analytics could help real institutions like RSSB monitor inclusion, identify vulnerable groups, and evaluate program reach, **without accessing confidential citizen data**

This simulation demonstrates how **AI and data engineering can serve social good**: by building safe, reproducible models that guide equity-centered policy design


* Created by Christa Rusanganwa Ingabire*
