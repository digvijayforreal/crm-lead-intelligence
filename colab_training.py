# ============================================================
# CRM Lead Intelligence System — ML Training Pipeline
# Run this in Google Colab, cell by cell
# ============================================================

# ── CELL 1: Install dependencies ────────────────────────────
# !pip install xgboost scikit-learn pandas numpy matplotlib seaborn


# ── CELL 2: Imports ─────────────────────────────────────────
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

print("All libraries imported successfully.")


# ── CELL 3: Upload and load dataset ─────────────────────────
# In Colab, run this to upload leads_db.csv from your computer:
#
#   from google.colab import files
#   files.upload()
#
# Then load it:

df = pd.read_csv("leads_db.csv")

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nConversion rate: {df['converted'].mean()*100:.1f}%")
print(f"\nIndustry breakdown:")
print(df.groupby("industry")["converted"].agg(["count", "sum", "mean"])
        .rename(columns={"count":"leads","sum":"converted","mean":"conv_rate"})
        .sort_values("conv_rate", ascending=False))


# ── CELL 4: Data exploration ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Calls vs conversion
df.boxplot(column="num_calls", by="converted", ax=axes[0])
axes[0].set_title("Calls by conversion")
axes[0].set_xlabel("Converted (0=No, 1=Yes)")

# Email opens vs conversion
df.boxplot(column="email_opens", by="converted", ax=axes[1])
axes[1].set_title("Email opens by conversion")
axes[1].set_xlabel("Converted (0=No, 1=Yes)")

# Website visits vs conversion
df.boxplot(column="website_visits", by="converted", ax=axes[2])
axes[2].set_title("Website visits by conversion")
axes[2].set_xlabel("Converted (0=No, 1=Yes)")

plt.suptitle("")
plt.tight_layout()
plt.savefig("feature_distribution.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: feature_distribution.png")


# ── CELL 5: Preprocessing ────────────────────────────────────
# Encode industry (categorical → numeric)
le = LabelEncoder()
df["industry_encoded"] = le.fit_transform(df["industry"])

print("Industry encoding:")
for i, cls in enumerate(le.classes_):
    print(f"  {cls} → {i}")

# Features and target
FEATURE_COLS = ["industry_encoded", "num_calls", "email_opens", "website_visits"]
X = df[FEATURE_COLS]
y = df["converted"]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target distribution:  {y.value_counts().to_dict()}")

# Train / test split (80 / 20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")


# ── CELL 6: Train XGBoost model ──────────────────────────────
model = XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)

model.fit(X_train, y_train)
print("XGBoost model trained successfully.")

# Cross-validation (5-fold)
cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
print(f"\nCross-validation ROC AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")


# ── CELL 7: Evaluate on test set ─────────────────────────────
y_pred  = model.predict(X_test)
y_prob  = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

print("=" * 45)
print("MODEL EVALUATION")
print("=" * 45)
print(f"\nROC AUC Score : {roc_auc:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Converted", "Converted"]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted No", "Predicted Yes"],
            yticklabels=["Actual No", "Actual Yes"])
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: confusion_matrix.png")


# ── CELL 8: ROC curve ────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — XGBoost Lead Scoring")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: roc_curve.png")


# ── CELL 9: Feature importance ───────────────────────────────
importance = pd.Series(
    model.feature_importances_,
    index=["Industry", "Num Calls", "Email Opens", "Website Visits"]
).sort_values()

plt.figure(figsize=(7, 4))
importance.plot(kind="barh", color="steelblue")
plt.title("Feature Importance — XGBoost")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120, bbox_inches="tight")
plt.show()
print("\nFeature importances:")
for feat, imp in importance.sort_values(ascending=False).items():
    print(f"  {feat:20s}: {imp:.4f}")


# ── CELL 10: Save model and encoder ──────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Saved: model.pkl")
print("Saved: label_encoder.pkl")

# Verify: reload and do a quick sanity check
with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

test_sample = np.array([[le.transform(["Technology"])[0], 7, 15, 40]])
score = loaded_model.predict_proba(test_sample)[0][1]
print(f"\nSanity check — Technology, 7 calls, 15 emails, 40 visits:")
print(f"  Conversion probability: {score:.4f}")
print(f"  Category: {'High' if score > 0.8 else 'Medium' if score > 0.5 else 'Low'}")

print("\nTraining complete. Download model.pkl and label_encoder.pkl.")


# ── CELL 11: Download files from Colab ──────────────────────
# Run this cell to download the trained files to your computer:
#
#   from google.colab import files
#   files.download("model.pkl")
#   files.download("label_encoder.pkl")
#   files.download("confusion_matrix.png")
#   files.download("roc_curve.png")
#   files.download("feature_importance.png")
