"""
Gaitly AI - Pediatric Gait Analysis Package
نظام الذكاء الاصطناعي ورؤية الحاسب لتحليل مشية وحركة الأطفال
Faculty of Computers and Information Technology, University of Tabuk
جامعة تبوك - كلية الحاسبات وتقنية المعلومات

هذه الحزمة البرمجية توفر الأدوات المعيارية لمعالجة الفيديوهات،
استخراج الوضعية ونقاط الهيكل العظمي، حساب المؤشرات البيوميكانيكية،
التصنيف الذكي عبر XGBoost، توليد تفسيرات SHAP، وتصدير التقارير السريرية.
"""

__version__ = "1.0.0"
__author__ = "Gaitly AI Team - University of Tabuk"

# Core Modules & Functions
from .pose_estimation import (
    get_pose_estimator,
    extract_landmarks_from_frame,
    get_landmark_coords,
    annotate_frame_with_skeleton,
    process_video_skeleton
)

from .feature_extraction import (
    calculate_joint_angle,
    calculate_symmetry_index,
    extract_frame_features,
    apply_butterworth_filter,
    process_and_save_gait_data,
    aggregate_video_features,
    generate_poc_dataset
)

from .model_inference import (
    GaitClassifier,
    train_and_evaluate,
    generate_shap_plots
)

from .report_generator import (
    generate_recommendations,
    generate_pdf_report
)

__all__ = [
    # Pose Estimation
    "get_pose_estimator",
    "extract_landmarks_from_frame",
    "get_landmark_coords",
    "annotate_frame_with_skeleton",
    "process_video_skeleton",
    # Feature Extraction
    "calculate_joint_angle",
    "calculate_symmetry_index",
    "extract_frame_features",
    "apply_butterworth_filter",
    "process_and_save_gait_data",
    "aggregate_video_features",
    "generate_poc_dataset",
    # Model & Inference
    "GaitClassifier",
    "train_and_evaluate",
    "generate_shap_plots",
    # Recommendations & Reporting
    "generate_recommendations",
    "generate_pdf_report"
]
