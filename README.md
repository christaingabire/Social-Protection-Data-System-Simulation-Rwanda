# Social Protection Data System Simulation — Rwanda

This project is a small, open-source simulation inspired by my work at the Rwanda Social Security Board. It models how data pipelines and analytics can help governments monitor coverage, identify inequalities, and design more inclusive social protection systems. In this project I'm entirely using **entirely synthetic data!**

### Overview

- Generates 100k+ rows of synthetic citizen-level data (age, gender, income, district, scheme type, etc.)
- Cleans and aggregates the data using a lightweight Python ETL pipeline
- Produces key indicators:
  - Enrollment rate by district  
  - Gender gap in coverage  
  - Urban–rural benefit disparities  
- Includes a small predictive model (logistic regression) to estimate who is least likely to be covered
- All visuals are generated automatically through the pipeline

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


### Tech Stack

Python · Pandas · Matplotlib · Seaborn · scikit-learn · Jupyter Notebook

### Why I Built It

Social protection data is often sensitive and fragmented. I wanted to show, in a transparent and ethical way, how open data systems could be designed to make coverage metrics visible and actionable. This small simulation captures that idea in a controlled, reproducible way

### Next Steps

- Add a database layer (SQLite or DuckDB) for versioned metrics  
- Explore an interactive dashboard (Streamlit or Superset)  
- Document lessons for **AI for Social Good** and **Data for Governance** work

---

* Created by Christa Rusanganwa Ingabire*
