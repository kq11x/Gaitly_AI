"""
================================================================================
Gaitly AI - Biomechanical Kinematics & Feature Extraction Module
وحدة استخراج الخصائص الحركية والتحليل البيوميكانيكي
================================================================================
Responsible for computing sagittal joint angles (Knee, Hip, Ankle), applying
Butterworth zero-phase low-pass filtering, calculating the Robinson Symmetry Index,
estimating cadence, and aggregating video-level feature vectors for ML classification.

مسؤولة عن:
1. حساب زوايا المفاصل السهمية (الركبة، الورك، الكاحل) باستخدام قانون جيب التمام.
2. تنقية الإشارات الحركية بمرشح بتروورث الرقمي من الدرجة الرابعة وتردد قطع 6 هرتز (Winter 2009).
3. حساب مؤشر تماثل روبنسون (Robinson Symmetry Index) لقياس التناظر بين الطرفين.
4. حساب معدل الخطوات (Cadence) عبر تقاطعات الإشارة الحركية.
5. تجميع المتجه المميز (5D Feature Vector) لتغذية نموذج XGBoost.

Academic References:
- Winter, D. A. (2009). Biomechanics and Motor Control of Human Movement (4th ed.). John Wiley & Sons.
- Robinson, R. O., Herzog, W., & Nigg, B. M. (1987). Use of force platform variables to quantify the effects of chiropractic manipulation on gait symmetry. JMPT.
================================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import os
from typing import Optional, Dict, Any, Union, List, Tuple

from .pose_estimation import get_pose_estimator, extract_landmarks_from_frame, get_landmark_coords


def calculate_joint_angle(
    A: Optional[Tuple[float, float]],
    B: Optional[Tuple[float, float]],
    C: Optional[Tuple[float, float]]
) -> Optional[float]:
    """
    Calculates the 2D planar angle at vertex B formed by segments BA and BC using the law of cosines:
    حساب الزاوية المستوية عند الرأس B المتشكل بالمتجهين BA و BC باستخدام قانون جيب التمام.

    Formula:
    --------
    u = A - B,   v = C - B
    cos(theta) = (u . v) / (||u|| * ||v||)
    theta = arccos(clip(cos(theta), -1.0, 1.0)) * (180 / pi)

    Anatomical Angle Mapping:
    -------------------------
    - Knee Angle: A = Hip (#23/#24), B = Knee (#25/#26), C = Ankle (#27/#28)
      (Full anatomical extension is ~170-180 degrees; flexion decreases this value)
    - Hip Angle : A = Shoulder (#11/#12), B = Hip (#23/#24), C = Knee (#25/#26)
    - Ankle Angle: A = Knee (#25/#26), B = Ankle (#27/#28), C = Heel (#29/#30)

    Parameters:
    -----------
    A, B, C : Optional[Tuple[float, float]]
        (x, y) normalized coordinates of the proximal, vertex, and distal keypoints.

    Returns:
    --------
    Optional[float] : Angle in degrees [0.0, 180.0], or None if any point is missing/collinear.
    """
    if A is None or B is None or C is None:
        return None

    u = np.array(A, dtype=np.float64) - np.array(B, dtype=np.float64)
    v = np.array(C, dtype=np.float64) - np.array(B, dtype=np.float64)

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    # Avoid zero division with numerical epsilon
    if norm_u < 1e-7 or norm_v < 1e-7:
        return None

    cos_theta = np.dot(u, v) / (norm_u * norm_v)
    # Clip to valid arccos domain [-1.0, 1.0] to prevent NaN on floating point drift
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_degrees = float(np.degrees(np.arccos(cos_theta)))
    return round(angle_degrees, 3)


def calculate_symmetry_index(
    left_val: Optional[float],
    right_val: Optional[float]
) -> Optional[float]:
    """
    Computes the Robinson Symmetry Index (SI) between homologous bilateral limbs:
    حساب مؤشر تماثل روبنسون (Robinson Symmetry Index) بين الطرفين الأيمن والأيسر.

    Formula:
    --------
    SI = (2 * |Left - Right| / (Left + Right)) * 100%

    Clinical Interpretation:
    ------------------------
    - SI <= 10.0% : Physiological Symmetry (نطاق طبيعي متماثل)
    - 10.0% < SI <= 18.0% : Mild to Moderate Asymmetry (تفاوت حركي خفيف إلى متوسط)
    - SI > 18.0% : Clinically Significant Asymmetry (عدم تناظر حركي واضح يستدعي تقييماً سريرياً)

    Parameters:
    -----------
    left_val : float or None
        Kinematic metric of the left limb (e.g. Mean Knee Angle).
    right_val : float or None
        Kinematic metric of the right limb.

    Returns:
    --------
    Optional[float] : Symmetry Index percentage [0.0, 200.0]%, or None if undefined.
    """
    if left_val is None or right_val is None:
        return None

    denominator = float(left_val + right_val)
    if abs(denominator) < 1e-6:
        return None

    si = (2.0 * abs(left_val - right_val)) / denominator * 100.0
    return round(float(si), 3)


def extract_frame_features(landmarks: Any) -> Dict[str, Optional[float]]:
    """
    Extracts bilateral sagittal joint angles and instantaneous symmetry indices from a single frame.
    استخراج زوايا المفاصل للطرفين السفليين ومؤشرات التماثل اللحظية لإطار واحد.

    Landmark Keypoints Extracted:
    ------------------------------
    - Left/Right Hip (23, 24)
    - Left/Right Knee (25, 26)
    - Left/Right Ankle (27, 28)
    - Left/Right Shoulder (11, 12)
    - Left/Right Heel (29, 30)
    """
    feature_keys = [
        "L_Hip", "L_Knee", "L_Ankle",
        "R_Hip", "R_Knee", "R_Ankle",
        "Knee_Symmetry", "Hip_Symmetry", "Ankle_Symmetry"
    ]

    if landmarks is None:
        return {k: None for k in feature_keys}

    # Extract 2D Landmark Coordinates
    L_hip = get_landmark_coords(landmarks, 23)
    R_hip = get_landmark_coords(landmarks, 24)
    L_knee = get_landmark_coords(landmarks, 25)
    R_knee = get_landmark_coords(landmarks, 26)
    L_ankle = get_landmark_coords(landmarks, 27)
    R_ankle = get_landmark_coords(landmarks, 28)
    L_shoulder = get_landmark_coords(landmarks, 11)
    R_shoulder = get_landmark_coords(landmarks, 12)
    L_heel = get_landmark_coords(landmarks, 29)
    R_heel = get_landmark_coords(landmarks, 30)

    features: Dict[str, Optional[float]] = {}

    # Sagittal Planar Joint Angles
    features["L_Knee"] = calculate_joint_angle(L_hip, L_knee, L_ankle)
    features["R_Knee"] = calculate_joint_angle(R_hip, R_knee, R_ankle)
    features["L_Hip"] = calculate_joint_angle(L_shoulder, L_hip, L_knee)
    features["R_Hip"] = calculate_joint_angle(R_shoulder, R_hip, R_knee)
    features["L_Ankle"] = calculate_joint_angle(L_knee, L_ankle, L_heel)
    features["R_Ankle"] = calculate_joint_angle(R_knee, R_ankle, R_heel)

    # Instantaneous Symmetry Indices
    features["Knee_Symmetry"] = calculate_symmetry_index(features["L_Knee"], features["R_Knee"])
    features["Hip_Symmetry"] = calculate_symmetry_index(features["L_Hip"], features["R_Hip"])
    features["Ankle_Symmetry"] = calculate_symmetry_index(features["L_Ankle"], features["R_Ankle"])

    return features


def apply_butterworth_filter(
    data_series: np.ndarray,
    fps: float,
    cutoff: float = 6.0,
    order: int = 4
) -> np.ndarray:
    """
    Applies a zero-phase 4th order Butterworth Low-Pass Filter (Winter, 2009 standard for human locomotion).
    تطبيق مرشح بتروورث الرقمي من الدرجة الرابعة (Zero-Phase Butterworth Filter) لإزالة الضوضاء الحركية.

    Why Winter's 6.0 Hz standard?
    -----------------------------
    Human voluntary gait dynamics are concentrated below 5-6 Hz. Higher frequencies represent
    camera jitter, clothing folds, or landmark tracking fluctuations. Zero-phase filtering
    (scipy.signal.filtfilt) avoids phase lag and preserves true gait event timings.

    Parameters:
    -----------
    data_series : np.ndarray
        Raw 1D time-series signal.
    fps : float
        Video sampling frame rate.
    cutoff : float
        Low-pass cutoff frequency in Hz (default: 6.0 Hz).
    order : int
        Filter order (default: 4).

    Returns:
    --------
    np.ndarray : Cleaned signal after forward and backward digital filtering.
    """
    series_clean = np.asarray(data_series, dtype=np.float64)

    # Impute remaining NaNs with linear interpolation/nearest value before filtering
    if np.isnan(series_clean).any():
        series_clean = pd.Series(series_clean).interpolate(method='linear', limit_direction='both').bfill().ffill().values

    # Insufficient length guard
    min_length = 3 * order + 1
    if len(series_clean) < min_length:
        return series_clean

    nyquist = 0.5 * fps
    if nyquist <= cutoff:
        # Cutoff must be less than Nyquist frequency
        normal_cutoff = 0.95
    else:
        normal_cutoff = min(cutoff / nyquist, 0.99)

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, series_clean)
    return filtered


def process_and_save_gait_data(
    video_path: str,
    output_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    Processes video frame-by-frame, extracts raw angles, interpolates missing values,
    applies the 4th-order Butterworth low-pass filter, and computes filtered symmetry timeseries.
    معالجة مقطع الفيديو إطاراً بإطار، استخراج الزوايا، تنقية الإشارات، وحفظ ملف CSV المنقى.

    Parameters:
    -----------
    video_path : str
        Source video filepath.
    output_csv : Optional[str]
        Destination CSV filepath (e.g., data/processed_csv/gait_features_clean.csv).

    Returns:
    --------
    pd.DataFrame containing filtered kinematic time-series.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    pose = get_pose_estimator()
    cap = cv2.VideoCapture(video_path)
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(raw_fps) if (raw_fps is not None and raw_fps > 0 and not np.isnan(raw_fps)) else 30.0

    raw_features: List[Dict[str, Any]] = []
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        landmarks = extract_landmarks_from_frame(frame, pose)
        feats = extract_frame_features(landmarks)
        feats["Frame"] = frame_idx
        feats["Timestamp_sec"] = frame_idx / fps
        raw_features.append(feats)
        frame_idx += 1

    cap.release()
    pose.close()

    if not raw_features:
        raise ValueError(f"No valid frames could be decoded from {video_path}")

    df_raw = pd.DataFrame(raw_features)

    cols_to_filter = [
        "L_Hip", "L_Knee", "L_Ankle",
        "R_Hip", "R_Knee", "R_Ankle",
        "Knee_Symmetry", "Hip_Symmetry", "Ankle_Symmetry"
    ]

    df_clean = df_raw.copy()
    for col in cols_to_filter:
        df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both').bfill().ffill()
        df_clean[f"{col}_Filtered"] = apply_butterworth_filter(df_clean[col].values, fps, cutoff=6.0)

    # Recompute symmetry from filtered angles for peak clinical accuracy
    df_clean["Knee_Symmetry_Filtered"] = df_clean.apply(
        lambda r: calculate_symmetry_index(r["L_Knee_Filtered"], r["R_Knee_Filtered"]), axis=1
    )
    df_clean["Hip_Symmetry_Filtered"] = df_clean.apply(
        lambda r: calculate_symmetry_index(r["L_Hip_Filtered"], r["R_Hip_Filtered"]), axis=1
    )
    df_clean["Ankle_Symmetry_Filtered"] = df_clean.apply(
        lambda r: calculate_symmetry_index(r["L_Ankle_Filtered"], r["R_Ankle_Filtered"]), axis=1
    )

    final_cols = [
        "Frame", "Timestamp_sec",
        "L_Knee_Filtered", "R_Knee_Filtered", "Knee_Symmetry_Filtered",
        "L_Hip_Filtered", "R_Hip_Filtered", "Hip_Symmetry_Filtered",
        "L_Ankle_Filtered", "R_Ankle_Filtered", "Ankle_Symmetry_Filtered"
    ]
    df_final = df_clean[final_cols]

    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df_final.to_csv(output_csv, index=False)

    return df_final


def aggregate_video_features(
    csv_or_df: Union[str, pd.DataFrame],
    subject_age: int = 7
) -> Dict[str, Any]:
    """
    Compiles full video kinematic series into a 5-dimensional feature vector:
    [Mean_KneeAngle_L, Mean_KneeAngle_R, Cadence, Symmetry_Knee, Age]
    تجميع المتجه المميز الخماسي للفيديو لتغذية نموذج التعلم الآلي.

    Cadence Estimation Formula:
    ---------------------------
    Cadence (steps/min) = (Zero Crossings / Duration in seconds) * 60
    Expected normative pediatric range (4-10 yrs): 90 - 120 steps/min.

    Parameters:
    -----------
    csv_or_df : Union[str, pd.DataFrame]
        CSV filepath or preloaded DataFrame.
    subject_age : int
        Chronological age of the child in years (target range: 4 - 10).

    Returns:
    --------
    Dict[str, Any] containing the 5 ML input features plus supplementary metadata.
    """
    if isinstance(csv_or_df, str):
        df = pd.read_csv(csv_or_df)
        source_name = os.path.basename(csv_or_df)
    else:
        df = csv_or_df
        source_name = "in_memory_dataframe"

    mean_knee_L = float(df["L_Knee_Filtered"].mean())
    mean_knee_R = float(df["R_Knee_Filtered"].mean())
    mean_symmetry = float(df["Knee_Symmetry_Filtered"].mean())

    # Cadence calculation via knee angle zero-crossings
    knee_signal = df["L_Knee_Filtered"].dropna().values
    cadence = 0.0

    if len(knee_signal) > 30 and "Timestamp_sec" in df.columns:
        duration_sec = float(df["Timestamp_sec"].iloc[-1] - df["Timestamp_sec"].iloc[0])
        if duration_sec >= 1.0:
            threshold = float(np.mean(knee_signal) - 0.3 * np.std(knee_signal))
            crossings = int(np.sum(np.diff(knee_signal < threshold).astype(int) == 1))
            cadence = float((crossings / duration_sec) * 60.0)
            # Clamp to plausible physiological range for children [30, 200]
            cadence = max(30.0, min(200.0, cadence))

    return {
        "Mean_KneeAngle_L": round(mean_knee_L, 2),
        "Mean_KneeAngle_R": round(mean_knee_R, 2),
        "Cadence": round(cadence, 1),
        "Symmetry_Knee": round(mean_symmetry, 2),
        "Age": int(subject_age),
        "Source_File": source_name
    }


def generate_poc_dataset(
    n_typical: int = 40,
    n_atypical: int = 40,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a proof-of-concept synthetic training dataset (80 samples) grounded
    in published pediatric gait literature baselines:
    توليد مجموعة بيانات تجريبية لإثبات المفهوم (80 عينة) مطابقة للدراسات البيوميكانيكية للأطفال:

    - Typical: Symmetry <= 7.5%, Cadence ~95-105, symmetric knee flexion angles (140-150 deg).
    - Atypical: Symmetry >= 12.0%, Cadence < 90 or > 120, asymmetric knee angles.

    Parameters:
    -----------
    n_typical : int (default: 40)
    n_atypical : int (default: 40)
    seed : int (default: 42)

    Returns:
    --------
    pd.DataFrame with 80 balanced samples and class 'Label' ("Typical" or "Atypical").
    """
    np.random.seed(seed)
    data = []

    # 1. Typical Class (40 samples)
    for _ in range(n_typical):
        base_angle = np.random.normal(145.0, 7.0)
        data.append({
            "Mean_KneeAngle_L": round(base_angle + np.random.normal(0, 1.8), 2),
            "Mean_KneeAngle_R": round(base_angle + np.random.normal(0, 1.8), 2),
            "Cadence": round(np.random.normal(102.0, 5.5), 1),
            "Symmetry_Knee": round(np.random.uniform(1.5, 7.5), 2),
            "Age": int(np.random.randint(4, 11)),
            "Label": "Typical"
        })

    # 2. Atypical Class (40 samples)
    for _ in range(n_atypical):
        knee_L = np.random.normal(134.0, 12.0)
        knee_R = np.random.normal(154.0, 12.0)
        data.append({
            "Mean_KneeAngle_L": round(knee_L, 2),
            "Mean_KneeAngle_R": round(knee_R, 2),
            "Cadence": round(np.random.normal(82.0, 14.0), 1),
            "Symmetry_Knee": round(np.random.uniform(12.0, 24.5), 2),
            "Age": int(np.random.randint(4, 11)),
            "Label": "Atypical"
        })

    return pd.DataFrame(data)
