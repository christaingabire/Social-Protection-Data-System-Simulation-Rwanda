import numpy as np
import pandas as pd
import random

np.random.seed(42)
random.seed(42)

# Rwanda's 30 districts
districts = [
    "Gasabo","Kicukiro","Nyarugenge","Musanze","Gakenke","Burera","Rulindo","Gicumbi","Huye","Nyaruguru","Nyamagabe","Gisagara","Ruhango","Nyanza","Kamonyi","Muhanga",
    "Rubavu","Rutsiro","Nyamasheke","Rusizi","Karongi","Ngororero","Nyabihu","Kayonza","Rwamagana","Ngoma","Kirehe","Bugesera","Gatsibo","Nyagatare"
]

genders = ["Female","Male"]
education = ["None","Primary","Secondary","Tertiary"]
income = ["Low","Lower-Middle","Upper-Middle","High"]
schemes = ["Pension","CommunityHealth","LongTermSavings"]

N = 100_000

# Generate synthetic citizens
df = pd.DataFrame({
    "person_id": np.arange(1, N+1),
    "age": np.random.randint(0, 90, N),
    "gender": np.random.choice(genders, N),
    "district": np.random.choice(districts, N),
    "urban_rural": np.random.choice(["Urban","Rural"], N, p=[0.3,0.7]),
    "education_level": np.random.choice(education, N, p=[0.1,0.4,0.35,0.15]),
    "income_level": np.random.choice(income, N, p=[0.25,0.35,0.25,0.15]),
    "scheme": np.random.choice(schemes, N, p=[0.4,0.5,0.1])
})

# Enrollment probability
def enroll_prob(row):
    base = 0.55
    if row["urban_rural"] == "Rural":
        base -= 0.1
    if row["education_level"] == "Tertiary":
        base += 0.1
    if row["income_level"] == "Low":
        base -= 0.1
    return base

df["enrolled"] = [np.random.rand() < enroll_prob(r) for _, r in df.iterrows()]

# Benefit amount: only for enrolled
df["benefit_amount"] = np.where(df["enrolled"]
, np.random.gamma(2, 5000, N), 0.0)

df.to_csv("data/synthetic_social_protection.csv", index=False)
print(" Synthetic data created -> data/synthetic_social_protection.csv")
