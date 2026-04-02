"""
Heart Disease Detection — ML Training Pipeline
BSc Final Year Project | CN6000

Run this FIRST to train the model and generate final_model.pkl
Then run: streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# PHASE 1 — LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────
def load_and_clean_data(file_path="heart.csv"):
    print("\n[PHASE 1] DATA LOADING & CLEANING")
    print("─" * 60)

    if not Path(file_path).exists():
        raise FileNotFoundError(f"'{file_path}' not found. Place it in the same folder.")

    df = pd.read_csv(file_path)
    df = df.replace('?', np.nan)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.median())
    df['target'] = (df['target'] > 0).astype(int)

    print(f"✓ Loaded '{file_path}'  |  {len(df)} rows  |  {df.shape[1]-1} features")
    print(f"  Class balance — Healthy: {(df['target']==0).sum()}  |  Disease: {(df['target']==1).sum()}")
    return df


# ─────────────────────────────────────────────────────────────────
# PHASE 2 — VISUALISATIONS
# ─────────────────────────────────────────────────────────────────
def generate_plots(df):
    print("\n[PHASE 2] GENERATING VISUALISATIONS")
    print("─" * 60)
    Path("plots").mkdir(exist_ok=True)

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.4)
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png', dpi=150)
    plt.close()

    # Class distribution
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x='target', data=df, palette=['#2ecc71', '#e74c3c'])
    ax.set_xticklabels(['Healthy (0)', 'Heart Disease (1)'])
    plt.title('Class Distribution')
    plt.tight_layout()
    plt.savefig('plots/target_distribution.png', dpi=150)
    plt.close()

    # Age distribution by class
    plt.figure(figsize=(8, 4))
    for label, color in [(0, '#2ecc71'), (1, '#e74c3c')]:
        subset = df[df['target'] == label]['age']
        subset.plot(kind='kde', label='Healthy' if label == 0 else 'Heart Disease', color=color)
    plt.title('Age Distribution by Class')
    plt.xlabel('Age')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/age_distribution.png', dpi=150)
    plt.close()

    print("✓ Plots saved → plots/")


# ─────────────────────────────────────────────────────────────────
# PHASE 3 — TRAIN & EVALUATE
# ─────────────────────────────────────────────────────────────────
def train_and_evaluate(df):
    print("\n[PHASE 3] TRAINING XGBOOST MODEL")
    print("─" * 60)

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')
    print(f"  5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Test set metrics
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(f"\n  Test Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision      : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall         : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1-Score       : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC        : {roc_auc_score(y_test, y_proba):.4f}")

    print("\n" + classification_report(y_test, y_pred,
                                       target_names=['Healthy', 'Heart Disease']))

    # Confusion matrix plot
    Path("plots").mkdir(exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Disease'],
                yticklabels=['Healthy', 'Disease'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=150)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#e74c3c', lw=2,
             label=f'AUC = {roc_auc_score(y_test, y_proba):.3f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve'); plt.legend()
    plt.tight_layout()
    plt.savefig('plots/roc_curve.png', dpi=150)
    plt.close()

    return pipeline


# ─────────────────────────────────────────────────────────────────
# PHASE 4 — SAVE MODEL
# ─────────────────────────────────────────────────────────────────
def save_model(pipeline, out="final_model.pkl"):
    print("\n[PHASE 4] SAVING MODEL")
    print("─" * 60)
    joblib.dump(pipeline, out)
    print(f"✓ Model saved → {out}")
    print("\n✅ DONE! Run your app with:  streamlit run app.py")


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_and_clean_data("heart.csv")
    generate_plots(df)
    pipeline = train_and_evaluate(df)
    save_model(pipeline)
