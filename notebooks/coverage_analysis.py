import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

df = pd.read_csv("data/synthetic_social_protection.csv")

# Convert categorical values to dummy variables
X = pd.get_dummies(df[["age","gender","urban_rural","education_level","income_level","scheme"]], drop_first=True)
y = df["enrolled"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
preds = model.predict_proba(X_test)[:,1]

print("ROC AUC:", roc_auc_score(y_test, preds))
print(classification_report(y_test, preds > 0.5))
