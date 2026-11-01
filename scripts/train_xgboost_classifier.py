"""
================================================================================
Gaitly AI - XGBoost Model Training and Cross-Validation Script
أداة تدريب نموذج XGBoost والتحقق المتقاطع وتوليد رسومات تفسير SHAP
================================================================================
Executes Stratified 5-Fold Cross Validation (NFR1 compliance),
evaluates quantitative metrics (Accuracy, Recall, Precision, F1),
generates Confusion Matrix and SHAP explainability charts,
and exports the serialized model artifact.

Academic Requirements Evaluated:
- NFR1: Accuracy >= 85%, Sensitivity / Recall >= 90%
- FR7: Supervised Classification with XGBoost
- FR8: Explainable AI (XAI) with SHAP TreeExplainer
================================================================================
"""

import os
import sys
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import shap

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Safe UTF-8 encoding on Windows terminal
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

FEATURE_COLUMNS = ["Mean_KneeAngle_L", "Mean_KneeAngle_R", "Cadence", "Symmetry_Knee", "Age"]


def train_xgboost(
    dataset_path: str = None,
    model_save_path: str = None,
    figures_dir: str = None,
    n_splits: int = 5,
    random_state: int = 42
):
    """
    Trains XGBoost model on the dataset with Stratified 5-Fold Cross-Validation,
    saves figures, and synchronizes artifacts across root and nested folders.
    """
    if dataset_path is None:
        dataset_path = PROJECT_ROOT / "data" / "processed_csv" / "pediatric_gait_poc_dataset.csv"
        if not dataset_path.exists():
            dataset_path = PROJECT_ROOT / "data" / "processed_csv" / "poc_training_dataset.csv"
        if not dataset_path.exists():
            dataset_path = PROJECT_ROOT / "Gaitly_AI" / "poc_training_dataset.csv"

    if model_save_path is None:
        model_save_path = PROJECT_ROOT / "models" / "gaitly_xgboost.joblib"

    if figures_dir is None:
        figures_dir = PROJECT_ROOT / "figures"

    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
    os.makedirs(str(figures_dir), exist_ok=True)

    print("=" * 70)
    print("🚀 [Gaitly AI] Training XGBoost Classifier with Stratified Cross-Validation")
    print(f"📂 Dataset Path : {dataset_path}")
    print(f"💾 Model Target : {model_save_path}")
    print(f"📊 Figures Dir  : {figures_dir}")
    print("=" * 70)

    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset file not found at {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    print(f"✅ Loaded {len(df)} samples from dataset:")
    print(df["Label"].value_counts())

    X = df[FEATURE_COLUMNS]
    y = (df["Label"] == "Atypical").astype(int)  # 1 = Atypical, 0 = Typical

    # 1. Stratified 5-Fold Cross Validation (NFR1 compliance)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=random_state
    )

    print("\n⏳ Executing Stratified 5-Fold Cross-Validation...")
    y_pred_cv = cross_val_predict(model, X, y, cv=skf)

    # Metrics calculation
    acc = accuracy_score(y, y_pred_cv)
    prec = precision_score(y, y_pred_cv)
    rec = recall_score(y, y_pred_cv)  # Sensitivity
    f1 = f1_score(y, y_pred_cv)

    print("\n" + "=" * 70)
    print("📈 Quantitative Validation Metrics (Evaluation against NFR1):")
    print("=" * 70)
    print(f"  • Accuracy    : {acc:.4f} ({acc*100:.1f}%)  [Target NFR1: >= 85%] -> {'✅ PASSED' if acc >= 0.85 else '⚠️ FAILED'}")
    print(f"  • Sensitivity : {rec:.4f} ({rec*100:.1f}%)  [Target NFR1: >= 90%] -> {'✅ PASSED' if rec >= 0.90 else '⚠️ FAILED'}")
    print(f"  • Precision   : {prec:.4f} ({prec*100:.1f}%)")
    print(f"  • F1-Score    : {f1:.4f} ({f1*100:.1f}%)")
    print("=" * 70)

    print("\n📋 Detailed Classification Report:")
    print(classification_report(y, y_pred_cv, target_names=["Typical", "Atypical"]))

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y, y_pred_cv)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=["Typical", "Atypical"],
        yticklabels=["Typical", "Atypical"],
        annot_kws={"size": 14}
    )
    plt.xlabel("Predicted Class", fontsize=12, fontweight='bold')
    plt.ylabel("Actual Ground Truth", fontsize=12, fontweight='bold')
    plt.title("Confusion Matrix (Stratified 5-Fold CV)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    cm_path = figures_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"💾 Saved Confusion Matrix to: {cm_path}")

    # 3. Fit on complete dataset and serialize model
    print("\n⏳ Fitting final model on all data and saving artifact...")
    model.fit(X, y)
    joblib.dump(model, str(model_save_path))
    print(f"✅ Serialized model successfully saved to: {model_save_path}")

    # Mirror model to nested Gaitly_AI folder if present
    nested_model = PROJECT_ROOT / "Gaitly_AI" / "gaitly_xgboost.joblib"
    if nested_model.parent.exists():
        joblib.dump(model, str(nested_model))

    # 4. Generate SHAP Explainability Plots (FR8 compliance)
    print("\n⏳ Calculating SHAP values for Explainable AI (XAI)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 4a. Summary Beeswarm Plot
    plt.figure()
    shap.summary_plot(shap_values, X, show=False, plot_size=(10, 6))
    plt.title("SHAP Feature Attribution (Beeswarm Summary)", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    summary_path = figures_dir / "shap_summary.png"
    plt.savefig(summary_path, dpi=200, bbox_inches='tight')
    plt.close()

    # 4b. Global Bar Importance Plot
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("Global Feature Importance (SHAP)", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    bar_path = figures_dir / "shap_importance.png"
    plt.savefig(bar_path, dpi=200, bbox_inches='tight')
    plt.close()

    # Mirror figures to nested Gaitly_AI folder if exists
    nested_dir = PROJECT_ROOT / "Gaitly_AI"
    if nested_dir.exists():
        try:
            shutil.copy2(str(cm_path), str(nested_dir / "confusion_matrix.png"))
            shutil.copy2(str(summary_path), str(nested_dir / "shap_summary.png"))
            shutil.copy2(str(bar_path), str(nested_dir / "shap_importance.png"))
        except Exception:
            pass

    # 4c. Per-sample CSV breakdown
    shap_csv_path = PROJECT_ROOT / "data" / "processed_csv" / "shap_feature_attributions.csv"
    feature_names = X.columns.tolist()
    explanations = []
    for i in range(len(X)):
        sample_shap = shap_values[i]
        sorted_idx = np.argsort(np.abs(sample_shap))[::-1]
        top = [(feature_names[j], sample_shap[j]) for j in sorted_idx[:3]]
        explanations.append({
            "Sample_Index": i,
            "Top_Feature_1": f"{top[0][0]} ({top[0][1]:+.3f})",
            "Top_Feature_2": f"{top[1][0]} ({top[1][1]:+.3f})",
            "Top_Feature_3": f"{top[2][0]} ({top[2][1]:+.3f})"
        })
    pd.DataFrame(explanations).to_csv(shap_csv_path, index=False)
    print(f"💾 Saved SHAP Plots & CSV: {summary_path}, {bar_path}")

    # 5. Inference Verification on Real Video Features
    real_video_csv = PROJECT_ROOT / "data" / "processed_csv" / "subject_video_feature_vector.csv"
    if not real_video_csv.exists():
        real_video_csv = PROJECT_ROOT / "data" / "processed_csv" / "real_video_features.csv"
    if not real_video_csv.exists():
        real_video_csv = PROJECT_ROOT / "Gaitly_AI" / "real_video_features.csv"

    if real_video_csv.exists():
        print("\n" + "=" * 70)
        print("🎬 Testing Inference on Real Video Kinematics:")
        print("=" * 70)
        df_real = pd.read_csv(real_video_csv)
        X_real = df_real[FEATURE_COLUMNS]
        pred = int(model.predict(X_real)[0])
        proba = model.predict_proba(X_real)[0]
        label = "Atypical" if pred == 1 else "Typical"
        print(f"  • Predicted Class : {label}")
        print(f"  • Confidence      : {proba[pred]:.2%}")
        print(f"  • Probabilities   : Typical={proba[0]:.2%}, Atypical={proba[1]:.2%}")
        print("=" * 70)

    print("\n🎉 XGBoost Training & Validation Completed Successfully!\n")


if __name__ == "__main__":
    train_xgboost()
