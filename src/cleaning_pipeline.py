import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
import seaborn as sns
from sqlalchemy import create_engine


# ---------- Load ----------
df = pd.read_csv("data/synthetic_social_protection.csv")

# ---------- Enforce dtypes (clean + academic) ----------
districts = [
    "Gasabo","Kicukiro","Nyarugenge","Musanze","Gakenke","Burera","Rulindo","Gicumbi",
    "Huye","Nyaruguru","Nyamagabe","Gisagara","Ruhango","Nyanza","Kamonyi","Muhanga",
    "Rubavu","Rutsiro","Nyamasheke","Rusizi","Karongi","Ngororero","Nyabihu",
    "Kayonza","Rwamagana","Ngoma","Kirehe","Bugesera","Gatsibo","Nyagatare"
]
cats = {
  "gender": ["Female","Male"],
  "urban_rural": ["Urban","Rural"],
  "education_level": ["None","Primary","Secondary","Tertiary"],
  "income_level": ["Low","Lower-Middle","Upper-Middle","High"],
  "scheme": ["Pension","CommunityHealth","LongTermSavings"]
}
df = df.astype({
    "person_id":"Int64",
    "age":"int16",
    "enrolled":"boolean",
    "benefit_amount":"float64"
})
# district + other categoricals
df["district"] = df["district"].astype(CategoricalDtype(districts))
for c, vals in cats.items():
    df[c] = df[c].astype(CategoricalDtype(vals))

# basic sanity checks
df = df[df["age"].between(0,120)]
df = df[df["benefit_amount"] >= 0]

# ---------- Aggregations ----------
# Enrollment rate by district
enroll_by_dist = df.groupby("district", observed=True)["enrolled"].mean().rename("enrollment_rate")

# Gender coverage by district
gender_cov = (
    df.groupby(["district","gender"], observed=True)["enrolled"]
      .mean().unstack(fill_value=float("nan"))
      .rename(columns={"Female":"coverage_female","Male":"coverage_male"})
)

# Gender gap = Male - Female
gender_cov["gender_gap"] = gender_cov["coverage_male"] - gender_cov["coverage_female"]

# Urban vs Rural average benefit among enrolled
urban_benefit = (
    df[df["enrolled"]]
      .groupby(["district","urban_rural"], observed=True)["benefit_amount"]
      .mean().unstack()
      .rename(columns={"Urban":"avg_benefit_urban","Rural":"avg_benefit_rural"})
)

# Final metrics table
metrics = (
    pd.concat([enroll_by_dist, gender_cov, urban_benefit], axis=1)
      .reset_index()
)

# ---------- Save ----------
os.makedirs("data", exist_ok=True)
metrics.to_csv("data/coverage_metrics.csv", index=False)
print(" coverage_metrics.csv updated")

# ---------- Charts ----------

sns.set(style="whitegrid")

# 1) Enrollment by district (horizontal bar)
plt.figure(figsize=(6,10))
sns.barplot(y="district", x="enrollment_rate",
            data=metrics.sort_values("enrollment_rate", ascending=True),
            palette="crest")
plt.title("Enrollment Rate by District")
plt.xlabel("Rate")
plt.ylabel("")
plt.tight_layout()
plt.savefig("figures/enrollment_rate_by_district.png", dpi=150)
plt.close()

# 2) Gender gap scatter plot
plt.figure(figsize=(8,5))
sns.scatterplot(data=metrics, x="coverage_male", y="coverage_female", hue="district", legend=False, s=80)
plt.plot([0,1],[0,1],"--",color="gray")
plt.title("Male vs Female Coverage Rate by District")
plt.xlabel("Male Coverage")
plt.ylabel("Female Coverage")
plt.tight_layout()
plt.savefig("figures/gender_gap_by_district.png", dpi=150)
plt.close()

# 3) Avg benefit: urban vs rural scatter
plt.figure(figsize=(8,5))
sns.scatterplot(data=metrics, x="avg_benefit_urban", y="avg_benefit_rural", hue="district", legend=False, s=80)
plt.plot([metrics["avg_benefit_urban"].min(), metrics["avg_benefit_urban"].max()],
         [metrics["avg_benefit_urban"].min(), metrics["avg_benefit_urban"].max()],
         "--", color="gray")
plt.title("Average Benefit: Urban vs Rural")
plt.xlabel("Urban")
plt.ylabel("Rural")
plt.tight_layout()
plt.savefig("figures/urban_vs_rural_benefit.png", dpi=150)
plt.close()

# 4) Benefit distribution histogram
plt.figure(figsize=(7,4))
sns.histplot(df[df["enrolled"]]["benefit_amount"], bins=30, kde=True, color="#1f77b4")
plt.title("Distribution of Benefit Amounts (Enrolled Only)")
plt.xlabel("Benefit (RWF)")
plt.tight_layout()
plt.savefig("figures/benefit_distribution.png", dpi=150)
plt.close()

print(" charts refreshed with new styles")


# ---------- Save to SQLite ----------
db_path = "data/social_protection.db"
engine = create_engine(f"sqlite:///{db_path}")

metrics.to_sql("coverage_metrics", engine, if_exists="replace", index=False)
print(f" Saved coverage_metrics to {db_path}")