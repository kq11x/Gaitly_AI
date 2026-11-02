"""
================================================================================
Gaitly AI - Machine Learning & Model Inference Module
وحدة التعلم الآلي والاستدلال وتفسير قرارات الذكاء الاصطناعي (XAI)
================================================================================
Handles XGBoost classifier training with Stratified 5-Fold Cross-Validation (NFR1 compliance),
model artifact serialization, real-time prediction, and SHAP explainability visualizations.

مسؤولة عن:
1. تحميل نموذج XGBoost المدرب من مسارات متعددة ومرنة.
2. التنبؤ بنمط المشية (Typical / Atypical) مع حساب درجات الثقة والاحتمالات.
3. تدريب النموذج والتحقق المتقاطع الطبقي خماسي الطيات (Stratified 5-Fold Cross Validation).
4. توليد رسومات وتفسيرات SHAP البيانية لتفسير أسباب التصنيف (Explainable AI - XAI).
5. توفير تفسير موضعي (Local SHAP Attribution) لكل عينة مريضة بشكل مستقل.
================================================================================
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional, List
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import shap

# Candidate default model artifact locations
DEFAULT_MODEL_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "gaitly_xgboost.joblib"),
    os.path.join("models", "gaitly_xgboost.joblib"),
    os.path.join(os.path.dirname(__file__), "..", "models", "gaitly_xgboost.joblib"),
    os.path.join("Gaitly_AI", "gaitly_xgboost.joblib"),
    "gaitly_xgboost.joblib"
]

FEATURE_COLUMNS = ["Mean_KneeAngle_L", "Mean_KneeAngle_R", "Cadence", "Symmetry_Knee", "Age"]


class GaitClassifier:
    """
    Production-ready wrapper for the Gaitly AI XGBoost gait pattern classifier.
    مغلّف جاهز للإنتاج لإجراء الاستدلال والتنبؤ وتوليد التفسيرات لحالات المشي.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[XGBClassifier] = None
        self.model_path = model_path
        self._explainer: Optional[shap.TreeExplainer] = None
        self._load_model()

    def _load_model(self):
        """Loads serialized XGBoost model artifact from candidate paths."""
        candidate_paths = [self.model_path] if self.model_path else DEFAULT_MODEL_PATHS
        for path in candidate_paths:
            if path and os.path.exists(path):
                try:
                    self.model = joblib.load(path)
                    self.model_path = path
                    self._explainer = shap.TreeExplainer(self.model)
                    return
                except Exception as e:
                    print(f"⚠️ Warning loading model from {path}: {e}")
        self.model = None

    def is_loaded(self) -> bool:
        """Returns True if the XGBoost model artifact is loaded and ready for inference."""
        return self.model is not None

    def predict(self, feature_vector: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes inference on a single 5D feature vector dictionary.
        إجراء التصنيف على متجه الخصائص وإرجاع النتيجة مع درجة الثقة والاحتمالات.

        Parameters:
        -----------
        feature_vector : Dict[str, Any]
            Dictionary containing keys:
            ['Mean_KneeAngle_L', 'Mean_KneeAngle_R', 'Cadence', 'Symmetry_Knee', 'Age']

        Returns:
        --------
        Dict[str, Any]:
            - label: "Typical" or "Atypical"
            - prediction_int: 0 (Typical) or 1 (Atypical)
            - confidence: float in [0.0, 1.0]
            - probabilities: {"Typical": p0, "Atypical": p1}
        """
        if self.model is None:
            raise RuntimeError(
                "Model is not loaded. Ensure 'models/gaitly_xgboost.joblib' exists or train a new model."
            )

        X = pd.DataFrame([feature_vector])[FEATURE_COLUMNS]
        pred_int = int(self.model.predict(X)[0])
        probas = self.model.predict_proba(X)[0]
        confidence = float(probas[pred_int])
        label = "Atypical" if pred_int == 1 else "Typical"

        return {
            "label": label,
            "prediction_int": pred_int,
            "confidence": round(confidence, 4),
            "probabilities": {
                "Typical": round(float(probas[0]), 4),
                "Atypical": round(float(probas[1]), 4)
            }
        }

    def explain_sample(self, feature_vector: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Calculates local SHAP feature attributions for a single individual patient.
        حساب قيم SHAP الموضعية للمريض لشرح أسباب التصنيف وتأثير كل متغير بيوميكانيكي.

        Returns:
        --------
        List of dicts sorted by absolute contribution:
        [{"feature": "Symmetry_Knee", "value": 15.2, "shap_value": +0.45, "effect": "Pushes towards Atypical"}]
        """
        if self.model is None:
            return []

        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)

        X = pd.DataFrame([feature_vector])[FEATURE_COLUMNS]
        shap_vals = self._explainer.shap_values(X)[0]

        attributions = []
        for feat, val, sv in zip(FEATURE_COLUMNS, X.iloc[0], shap_vals):
            effect = "Pushes towards Atypical (غير نمطية)" if sv > 0 else "Pushes towards Typical (نمطية)"
            attributions.append({
                "feature": feat,
                "value": round(float(val), 2),
                "shap_value": round(float(sv), 4),
                "abs_importance": round(abs(float(sv)), 4),
                "effect": effect
            })

        attributions.sort(key=lambda item: item["abs_importance"], reverse=True)
        return attributions


def train_and_evaluate(
    dataset_path: str,
    model_save_path: str = "models/gaitly_xgboost.joblib",
    figures_dir: str = "figures",
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[XGBClassifier, Dict[str, float]]:
    """
    Trains XGBoost classifier on dataset using 5-Fold Stratified Cross-Validation (NFR1 compliance).
    تدريب نموذج XGBoost مع التحقق المتقاطع 5-Fold وتصدير مصفوفة الالتباس والمقاييس الكمية.

    Validation Targets (Graduation Project NFR1):
    --------------------------------------------
    - Accuracy >= 85.0%
    - Recall / Sensitivity >= 90.0%
    """
    df = pd.read_csv(dataset_path)
    X = df[FEATURE_COLUMNS]
    y = (df["Label"] == "Atypical").astype(int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=random_state
    )

    y_pred_cv = cross_val_predict(model, X, y, cv=skf)

    # Fit final production model on the complete cohort
    model.fit(X, y)
    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
    joblib.dump(model, model_save_path)

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred_cv)),
        "precision": float(precision_score(y, y_pred_cv)),
        "recall": float(recall_score(y, y_pred_cv)),
        "f1": float(f1_score(y, y_pred_cv))
    }

    # Confusion matrix plot
    os.makedirs(figures_dir, exist_ok=True)
    cm = confusion_matrix(y, y_pred_cv)
    plt.figure(figsize=(6, 4.5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=["Typical", "Atypical"],
        yticklabels=["Typical", "Atypical"],
        annot_kws={"size": 13}
    )
    plt.xlabel("Predicted Pattern", fontsize=11, fontweight='bold')
    plt.ylabel("Actual Ground Truth", fontsize=11, fontweight='bold')
    plt.title("Confusion Matrix (Stratified 5-Fold CV)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(figures_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    return model, metrics


def generate_shap_plots(
    model: XGBClassifier,
    X: pd.DataFrame,
    figures_dir: str = "figures",
    csv_output_path: Optional[str] = "data/processed_csv/shap_per_sample.csv"
):
    """
    Generates SHAP summary plot, bar feature importance plot, and per-sample explanation CSV.
    توليد مخططات SHAP التفسيرية (Beeswarm ومخطط الأهمية العامة) وحفظ قيم التفسير في ملف CSV.
    """
    os.makedirs(figures_dir, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 1. Beeswarm summary
    plt.figure()
    shap.summary_plot(shap_values, X, show=False, plot_size=(10, 6))
    plt.title("SHAP Feature Importance (Beeswarm Summary)", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "shap_summary.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # 2. Global Bar Importance
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("Global Feature Importance (Mean |SHAP|)", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "shap_importance.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # 3. CSV breakdown per sample
    if csv_output_path:
        os.makedirs(os.path.dirname(os.path.abspath(csv_output_path)), exist_ok=True)
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
        pd.DataFrame(explanations).to_csv(csv_output_path, index=False)
