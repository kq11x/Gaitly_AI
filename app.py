"""
================================================================================
Gaitly AI - Interactive Clinical Web Dashboard (Streamlit)
لوحة التحكم التفاعلية الشاملة لنظام تحليل مشية الأطفال بالذكاء الاصطناعي
================================================================================
Tabuk University - Faculty of Computers & Information Technology
جامعة تبوك - كلية الحاسبات وتقنية المعلومات
مشروع تخرج: نظام ذكي لتحليل مشية وحركة الأطفال (Gaitly AI)
إشراف: د. جوهرة الجابري | العام الجامعي: 2026/2027

Features:
- Tab 1: Video Upload & Real-Time Pose Estimation
- Tab 2: Biomechanical Kinematics & Polar Radar Symmetry Chart
- Tab 3: XGBoost AI Classification & SHAP Explainability (XAI)
- Tab 4: Evidence-Based Clinical Recommendations & Formal PDF Export
================================================================================
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.signal import butter, filtfilt
import joblib
import tempfile
import os
from datetime import datetime
from pathlib import Path

# ==========================================
# 1. إعدادات الصفحة وهوية النظام
# ==========================================
st.set_page_config(
    page_title="Gaitly AI - Pediatric Gait Analysis System",
    page_icon="🚶‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تهيئة متغيرات الجلسة (Session State)
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False
if "features" not in st.session_state:
    st.session_state["features"] = None
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None
if "confidence" not in st.session_state:
    st.session_state["confidence"] = None
if "recommendations" not in st.session_state:
    st.session_state["recommendations"] = None
if "csv_path" not in st.session_state:
    st.session_state["csv_path"] = None

# ==========================================
# 2. أنماط CSS المخصصة للواجهة الاحترافية
# ==========================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.1rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        margin: 0.6rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .warning-box {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #856404;
    }
    .success-box {
        background: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #155724;
    }
    .disclaimer {
        background: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1.1rem;
        border-radius: 8px;
        margin: 1.2rem 0;
        font-size: 0.92em;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. الدوال البيوميكانيكية الأساسية
# ==========================================

def calculate_joint_angle(A, B, C):
    """
    حساب الزاوية المستوية عند الرأس B المتشكل بالمتجهين BA و BC باستخدام قانون جيب التمام:
    cos(theta) = (u . v) / (||u|| * ||v||)
    """
    if A is None or B is None or C is None:
        return None
    u = np.array(A, dtype=np.float64) - np.array(B, dtype=np.float64)
    v = np.array(C, dtype=np.float64) - np.array(B, dtype=np.float64)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u < 1e-6 or norm_v < 1e-6:
        return None
    cos_theta = np.clip(np.dot(u, v) / (norm_u * norm_v), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def get_landmark_coords(landmarks, landmark_id, visibility_threshold=0.5):
    """استخراج الإحداثيات (x, y) لنقطة هيكلية إذا كانت درجة الظهور أعلى من العتبة."""
    if landmarks is None or landmark_id >= len(landmarks.landmark):
        return None
    lm = landmarks.landmark[landmark_id]
    return (lm.x, lm.y) if getattr(lm, "visibility", 1.0) >= visibility_threshold else None


def calculate_symmetry_index(left_val, right_val):
    """
    حساب مؤشر تماثل روبنسون (Robinson Symmetry Index):
    SI = (2 * |Left - Right| / (Left + Right)) * 100%
    """
    if left_val is None or right_val is None:
        return None
    denominator = float(left_val + right_val)
    if abs(denominator) < 1e-6:
        return None
    return float((2.0 * abs(left_val - right_val)) / denominator * 100.0)


def extract_frame_features(landmarks):
    """استخراج الزوايا السهمية ومؤشرات التماثل اللحظية لإطار مفرد."""
    features = {}
    if landmarks is None:
        return {k: None for k in [
            "L_Hip", "L_Knee", "L_Ankle", "R_Hip", "R_Knee", "R_Ankle",
            "Knee_Symmetry", "Hip_Symmetry", "Ankle_Symmetry"
        ]}

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

    features["L_Knee"] = calculate_joint_angle(L_hip, L_knee, L_ankle)
    features["R_Knee"] = calculate_joint_angle(R_hip, R_knee, R_ankle)
    features["L_Hip"] = calculate_joint_angle(L_shoulder, L_hip, L_knee)
    features["R_Hip"] = calculate_joint_angle(R_shoulder, R_hip, R_knee)
    features["L_Ankle"] = calculate_joint_angle(L_knee, L_ankle, L_heel)
    features["R_Ankle"] = calculate_joint_angle(R_knee, R_ankle, R_heel)

    features["Knee_Symmetry"] = calculate_symmetry_index(features["L_Knee"], features["R_Knee"])
    features["Hip_Symmetry"] = calculate_symmetry_index(features["L_Hip"], features["R_Hip"])
    features["Ankle_Symmetry"] = calculate_symmetry_index(features["L_Ankle"], features["R_Ankle"])
    return features


def apply_butterworth_filter(data_series, fps, cutoff=6.0, order=4):
    """تطبيق مرشح بتروورث الرقمي من الدرجة الرابعة (Winter 2009 standard, Cutoff = 6.0 Hz)."""
    series_clean = pd.Series(data_series).interpolate(method='linear', limit_direction='both').bfill().ffill().values
    if len(series_clean) <= (3 * order):
        return series_clean
    nyquist = 0.5 * fps
    normal_cutoff = min(cutoff / nyquist, 0.99)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, series_clean)


def process_and_save_gait_data(video_path, output_csv):
    """معالجة الفيديو إطاراً بإطار، تنقية الإشارات الحركية وتصدير السلسلة الزمنية المنقاة."""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(raw_fps) if (raw_fps is not None and raw_fps > 0 and not np.isnan(raw_fps)) else 30.0

    raw_features = []
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_feats = extract_frame_features(results.pose_landmarks)
        frame_feats["Frame"] = frame_idx
        frame_feats["Timestamp_sec"] = frame_idx / fps
        raw_features.append(frame_feats)
        frame_idx += 1

    cap.release()
    pose.close()

    if not raw_features:
        raise ValueError(f"Could not extract frames from video: {video_path}")

    df_raw = pd.DataFrame(raw_features)

    cols_to_filter = [
        "L_Hip", "L_Knee", "L_Ankle", "R_Hip", "R_Knee", "R_Ankle",
        "Knee_Symmetry", "Hip_Symmetry", "Ankle_Symmetry"
    ]

    df_clean = df_raw.copy()
    for col in cols_to_filter:
        df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both').bfill().ffill()
        df_clean[f"{col}_Filtered"] = apply_butterworth_filter(df_clean[col].values, fps, cutoff=6.0)

    # إعادة حساب مؤشرات التماثل من الزوايا المنقاة لضمان أعلى دقة
    df_clean["Knee_Symmetry_Filtered"] = df_clean.apply(
        lambda row: calculate_symmetry_index(row["L_Knee_Filtered"], row["R_Knee_Filtered"]), axis=1
    )
    df_clean["Hip_Symmetry_Filtered"] = df_clean.apply(
        lambda row: calculate_symmetry_index(row["L_Hip_Filtered"], row["R_Hip_Filtered"]), axis=1
    )
    df_clean["Ankle_Symmetry_Filtered"] = df_clean.apply(
        lambda row: calculate_symmetry_index(row["L_Ankle_Filtered"], row["R_Ankle_Filtered"]), axis=1
    )

    final_cols = [
        "Frame", "Timestamp_sec",
        "L_Knee_Filtered", "R_Knee_Filtered", "Knee_Symmetry_Filtered",
        "L_Hip_Filtered", "R_Hip_Filtered", "Hip_Symmetry_Filtered",
        "L_Ankle_Filtered", "R_Ankle_Filtered", "Ankle_Symmetry_Filtered"
    ]

    df_final = df_clean[final_cols]
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_final.to_csv(output_csv, index=False)
    return df_final


def aggregate_video_features(csv_path, subject_age=7):
    """تجميع المتجه المميز الخماسي للمقطع المسجل وحساب معدل الخطوات (Cadence)."""
    df = pd.read_csv(csv_path)

    mean_knee_L = float(df["L_Knee_Filtered"].mean())
    mean_knee_R = float(df["R_Knee_Filtered"].mean())
    mean_symmetry = float(df["Knee_Symmetry_Filtered"].mean())

    knee_signal = df["L_Knee_Filtered"].dropna().values
    cadence = 0.0
    if len(knee_signal) > 30 and "Timestamp_sec" in df.columns:
        duration_sec = float(df["Timestamp_sec"].iloc[-1] - df["Timestamp_sec"].iloc[0])
        if duration_sec >= 1.0:
            threshold = float(np.mean(knee_signal) - 0.3 * np.std(knee_signal))
            crossings = int(np.sum(np.diff(knee_signal < threshold).astype(int) == 1))
            cadence = float((crossings / duration_sec) * 60.0)
            cadence = max(30.0, min(200.0, cadence))

    return {
        "Mean_KneeAngle_L": round(mean_knee_L, 2),
        "Mean_KneeAngle_R": round(mean_knee_R, 2),
        "Cadence": round(cadence, 1),
        "Symmetry_Knee": round(mean_symmetry, 2),
        "Age": int(subject_age),
        "Source_File": os.path.basename(csv_path)
    }


def generate_recommendations(feature_vector: dict, prediction: str, confidence: float) -> dict:
    """محرك التوصيات الإرشادية المبني على المعايير البيوميكانيكية للأطفال."""
    recommendations = []
    clinical_guidance_en = []
    flags = []

    symm = feature_vector.get("Symmetry_Knee", 0.0)
    if symm > 18.0:
        flags.append(f"⚠️ عدم تماثل حركي واضح بين الطرفين (مؤشر روبنسون: {symm:.1f}% > 18%)")
        recommendations.append("يُنصح بإجراء فحص سريري تخصصي لدى طبيب عظام أطفال أو أخصائي علاج طبيعي معتمد لتقييم التوازن الحركي.")
        clinical_guidance_en.append(f"Pronounced lower limb kinematic asymmetry (Robinson Index: {symm:.1f}%). Clinical gait evaluation recommended.")
    elif symm > 10.0:
        flags.append(f"⚠️ تفاوت حركي خفيف إلى متوسط بين الطرفين ({symm:.1f}%)")
        recommendations.append("يُنصح بملاحظة حركة الطفل أثناء الجري وصعود السلالم وممارسة الأنشطة اليومية.")
        clinical_guidance_en.append(f"Mild to moderate limb asymmetry (Robinson Index: {symm:.1f}%). Continue functional monitoring.")

    cadence = feature_vector.get("Cadence", 0.0)
    if cadence > 0.0:
        if cadence < 90.0:
            flags.append(f"⚠️ معدل الخطوات أقل من النطاق الطبيعي المتوقع ({cadence:.1f} خطوة/دقيقة - الطبيعي: 90-120)")
            recommendations.append("يُنصح بمراقبة سرعة المشي والتأكد من عدم وجود آلام عضلية أو إجهاد سريع.")
            clinical_guidance_en.append(f"Reduced cadence ({cadence:.1f} steps/min). Assess endurance and joint comfort.")
        elif cadence > 120.0:
            flags.append(f"⚠️ معدل خطوات مرتفع نسبياً ({cadence:.1f} خطوة/دقيقة)")
            recommendations.append("يُنصح بالتحقق من طول الخطوة (Stride Length) وثبات التوازن الديناميكي.")
            clinical_guidance_en.append(f"Elevated cadence ({cadence:.1f} steps/min). May indicate compensatory rapid stepping.")

    angle_diff = abs(feature_vector.get("Mean_KneeAngle_L", 0.0) - feature_vector.get("Mean_KneeAngle_R", 0.0))
    if angle_diff > 15.0:
        flags.append(f"⚠️ تباين ملحوظ في متوسط زوايا ثني الركبة بين الطرفين ({angle_diff:.1f} درجة)")
        recommendations.append("يُنصح بتقييم المدى الحركي للمفاصل (ROM) وتوازن القوة العضلية بين الجانبين.")
        clinical_guidance_en.append(f"Sagittal knee angle discrepancy ({angle_diff:.1f} deg). Evaluate Range of Motion.")

    if not flags:
        flags.append("✅ لا توجد مؤشرات مقلقة؛ كافة القياسات تقع ضمن النطاق الطبيعي المتوقع")
        recommendations.append("المؤشرات الحركية والزوايا والتماثل تقع ضمن النطاق العمري الطبيعي. يُنصح بمتابعة النمو الحركي بشكل دوري.")
        clinical_guidance_en.append("Kinematic metrics conform to developmental baseline ranges. Routine monitoring recommended.")

    return {
        "Prediction": prediction,
        "Confidence": f"{confidence:.2%}" if isinstance(confidence, float) else str(confidence),
        "Flags": flags,
        "Recommendations": recommendations,
        "Clinical_Guidance_EN": clinical_guidance_en,
        "Disclaimer": "هذه النتائج والتوصيات صادرة عن أداة ذكاء اصطناعي مساعدة للمسح الاسترشادي المبكر فقط، ولا تُعد بديلاً عن الفحص والتشخيص الطبي السريري المتخصص."
    }

# ==========================================
# 4. تحميل نموذج الذكاء الاصطناعي (XGBoost)
# ==========================================
@st.cache_resource
def load_model():
    """البحث عن النموذج المدرب في مسارات متعددة وتحميله."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(base_dir, "models", "gaitly_xgboost.joblib"),
        os.path.join(base_dir, "Gaitly_AI", "gaitly_xgboost.joblib"),
        os.path.join("models", "gaitly_xgboost.joblib"),
        os.path.join("Gaitly_AI", "gaitly_xgboost.joblib"),
        "gaitly_xgboost.joblib"
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception:
                pass
    return None

# ==========================================
# 5. الشريط الجانبي (Sidebar)
# ==========================================
st.sidebar.title("🚶‍♂️ Gaitly AI")
st.sidebar.markdown("---")
st.sidebar.markdown("**Pediatric Movement & Gait Analysis**")
st.sidebar.markdown("**الفئة المستهدفة:** الأطفال من 4 إلى 10 سنوات")
st.sidebar.markdown("**الإصدار:** 1.0.0 (الدفعة الأكاديمية 2026/2027)")
st.sidebar.markdown("**جامعة تبوك - كلية الحاسبات**")
st.sidebar.markdown("---")
st.sidebar.info("💡 قم برفع فيديو المشي الجانبي (Sagittal View) لبدء التحليل الفوري.")

# ==========================================
# 6. الترويسة الرئيسية
# ==========================================
st.markdown("""
<div class="main-header">
    <h1>🚶‍♂️ نظام Gaitly AI</h1>
    <h3>تحليل مشية وحركة الأطفال بالذكاء الاصطناعي ورؤية الحاسب</h3>
    <p>Markerless • Explainable (SHAP) • Child-Focused Biomechanics</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 7. التبويبات الرئيسية الأربعة
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📹 رفع ومعالجة الفيديو (Upload & Analyze)",
    "📊 التحليل البيوميكانيكي (Biomechanical Kinematics)",
    "🤖 تصنيف الذكاء الاصطناعي (AI & SHAP XAI)",
    "💡 التوصيات وتصدير التقرير (Recommendations & PDF)"
])

# -----------------------------------------------------------------------------
# التبويب 1: رفع الفيديو واستخراج الوضعية
# -----------------------------------------------------------------------------
with tab1:
    st.header("📹 رفع الفيديو واستخراج الهيكل العظمي")
    col_input, col_preview = st.columns([1, 2])

    with col_input:
        uploaded_file = st.file_uploader(
            "اختر فيديو المشي للطفل (Sagittal Walking Video)",
            type=['mp4', 'mov', 'avi'],
            help="يُفضل فيديو مسجل بدقة جيدة وبمعدل 30 إطار/ثانية على الأقل."
        )

        subject_age = st.slider(
            "عمر الطفل (بالسنوات)",
            min_value=4, max_value=10, value=7,
            help="الفئة العمرية المستهدفة في النموذج: 4 - 10 سنوات"
        )

        analyze_btn = st.button("🔍 بدء التحليل الحركي والذكي", type="primary", use_container_width=True)

    with col_preview:
        if uploaded_file is not None:
            uploaded_file.seek(0)
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name

            st.video(uploaded_file)

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = (total_frames / fps) if fps > 0 else 0
            cap.release()

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("مدة المقطع", f"{duration:.1f} ثانية")
            col_m2.metric("معدل الإطارات", f"{fps:.1f} FPS")
            col_m3.metric("إجمالي الإطارات", f"{total_frames}")

    # معالجة الفيديو عند الضغط على زر التحليل
    if analyze_btn and uploaded_file is not None:
        with st.spinner("🔬 جاري استخراج نقاط الهيكل العظمي (MediaPipe Pose) وحساب الزوايا البيوميكانيكية..."):
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = os.path.join(tempfile.gettempdir(), f"gaitly_features_{timestamp_str}.csv")

            process_and_save_gait_data(video_path, csv_path)
            features = aggregate_video_features(csv_path, subject_age=subject_age)

            model = load_model()
            if model is None:
                st.error("❌ لم يتم العثور على ملف النموذج المدرب gaitly_xgboost.joblib.")
            else:
                feature_cols = ["Mean_KneeAngle_L", "Mean_KneeAngle_R", "Cadence", "Symmetry_Knee", "Age"]
                X_input = pd.DataFrame([features])[feature_cols]

                prediction_val = int(model.predict(X_input)[0])
                probas = model.predict_proba(X_input)[0]
                label = "Atypical" if prediction_val == 1 else "Typical"
                confidence = float(probas[prediction_val])

                recommendations = generate_recommendations(features, label, confidence)

                # حفظ النتائج في Session State
                st.session_state["analysis_done"] = True
                st.session_state["features"] = features
                st.session_state["prediction"] = label
                st.session_state["confidence"] = confidence
                st.session_state["recommendations"] = recommendations
                st.session_state["csv_path"] = csv_path

                st.success("✅ اكتمل تحليل الفيديو واستخراج المؤشرات بنجاح! انتقل إلى التبويبات التالية للاطلاع على النتائج.")
                st.rerun()

# -----------------------------------------------------------------------------
# التبويب 2: التحليل البيوميكانيكي ورادار التماثل
# -----------------------------------------------------------------------------
with tab2:
    st.header("📊 التحليل البيوميكانيكي للمفاصل الحركية")

    if st.session_state.get("analysis_done", False):
        csv_path = st.session_state["csv_path"]
        df = pd.read_csv(csv_path)

        st.subheader("🦵 زوايا ثني الركبة عبر الزمن (Knee Flexion Kinematics)")
        fig1, ax1 = plt.subplots(figsize=(11, 3.8))
        ax1.plot(df["Timestamp_sec"], df["L_Knee_Filtered"], label="Left Knee (الركبة اليسرى)", color="#2a5298", linewidth=2)
        ax1.plot(df["Timestamp_sec"], df["R_Knee_Filtered"], label="Right Knee (الركبة اليمنى)", color="#dc3545", linewidth=2)
        ax1.set_xlabel("Time (seconds) - الزمن بالثواني", fontsize=10)
        ax1.set_ylabel("Knee Angle (degrees) - الزاوية بالدرجات", fontsize=10)
        ax1.set_title("Sagittal Knee Flexion Angles (Winter 2009 Filtered)", fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(df["Timestamp_sec"].min(), df["Timestamp_sec"].max())
        st.pyplot(fig1)
        plt.close(fig1)

        col_hip, col_ankle = st.columns(2)
        with col_hip:
            st.subheader("🦴 زوايا الورك (Hip Angles)")
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.plot(df["Timestamp_sec"], df["L_Hip_Filtered"], label="Left Hip", color="#2a5298")
            ax2.plot(df["Timestamp_sec"], df["R_Hip_Filtered"], label="Right Hip", color="#dc3545")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Angle (°)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            plt.close(fig2)

        with col_ankle:
            st.subheader("🦶 زوايا الكاحل (Ankle Angles)")
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            ax3.plot(df["Timestamp_sec"], df["L_Ankle_Filtered"], label="Left Ankle", color="#2a5298")
            ax3.plot(df["Timestamp_sec"], df["R_Ankle_Filtered"], label="Right Ankle", color="#dc3545")
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Angle (°)")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)
            plt.close(fig3)

        st.markdown("---")
        st.subheader("⚖️ تحليل التماثل الحركي بين الطرفين (Robinson Symmetry Index)")

        symmetry_data = {
            "Knee": df["Knee_Symmetry_Filtered"].mean(),
            "Hip": df["Hip_Symmetry_Filtered"].mean(),
            "Ankle": df["Ankle_Symmetry_Filtered"].mean()
        }

        col_radar, col_metrics = st.columns([1, 1])

        with col_radar:
            fig_radar, ax_radar = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
            categories = list(symmetry_data.keys())
            values = list(symmetry_data.values())
            N = len(categories)

            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            values += values[:1]
            angles += angles[:1]

            ax_radar.plot(angles, values, 'o-', linewidth=2, color='#2a5298')
            ax_radar.fill(angles, values, alpha=0.25, color='#2a5298')
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(categories, fontsize=11)
            ax_radar.set_ylim(0, max(max(values) * 1.3, 25))
            ax_radar.set_title("Robinson Symmetry Radar (%)", fontsize=12, fontweight='bold', pad=15)
            ax_radar.axhline(y=10, color='red', linestyle='--', linewidth=1.5, label='Normal Threshold (10%)')
            ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
            ax_radar.grid(True)
            st.pyplot(fig_radar)
            plt.close(fig_radar)

        with col_metrics:
            st.markdown("#### ملخص مؤشرات التماثل:")
            for joint, val in symmetry_data.items():
                is_normal = val <= 10.0
                status_icon = "✅" if is_normal else "⚠️"
                delta_str = f"{val - 10.0:+.1f}% عن العتبة الطبيعية (10%)" if val > 10 else "ضمن النطاق المتماثل"
                st.metric(
                    label=f"{status_icon} تماثل {joint} ({joint} Symmetry)",
                    value=f"{val:.2f}%",
                    delta=delta_str if val > 10 else None,
                    delta_color="inverse"
                )
    else:
        st.info("👆 يرجى رفع مقطع الفيديو وبدء التحليل في التبويب الأول أولاً.")

# -----------------------------------------------------------------------------
# التبويب 3: تصنيف الذكاء الاصطناعي وقابلية التفسير (SHAP)
# -----------------------------------------------------------------------------
with tab3:
    st.header("🤖 تصنيف الذكاء الاصطناعي وتفسير القرارات (XAI)")

    if st.session_state.get("analysis_done", False):
        prediction = st.session_state["prediction"]
        confidence = st.session_state["confidence"]
        features = st.session_state["features"]

        if prediction == "Atypical":
            st.markdown(f"""
            <div class="warning-box">
                <h2>⚠️ نمط المشية المصنف: <strong>غير نمطي (Atypical)</strong></h2>
                <p><strong>درجة ثقة النموذج (Confidence):</strong> {confidence:.2%}</p>
                <p>يُظهر النمط الحركي مؤشرات بيوميكانيكية تحيد عن المعايير النمائية المتوقعة لعمر {features['Age']} سنوات.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-box">
                <h2>✅ نمط المشية المصنف: <strong>نمطي وطبيعي (Typical)</strong></h2>
                <p><strong>درجة ثقة النموذج (Confidence):</strong> {confidence:.2%}</p>
                <p>تتوافق مؤشرات الزوايا والتماثل مع النطاق النمائي الطبيعي للأطفال لعمر {features['Age']} سنوات.</p>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("📋 المتجه المميز المستخرج (Extracted Feature Vector)")
        feat_display = pd.DataFrame([{
            "متوسط زاوية الركبة اليسرى (Mean Knee L)": f"{features['Mean_KneeAngle_L']}°",
            "متوسط زاوية الركبة اليمنى (Mean Knee R)": f"{features['Mean_KneeAngle_R']}°",
            "معدل الخطوات (Cadence)": f"{features['Cadence']} خطوة/دقيقة",
            "مؤشر تماثل الركبة (Symmetry Knee)": f"{features['Symmetry_Knee']}%",
            "عمر الطفل (Age)": f"{features['Age']} سنوات"
        }]).T
        feat_display.columns = ["القيمة المستخرجة"]
        st.dataframe(feat_display, use_container_width=True)

        # عرض مخططات SHAP التفسيرية
        base_dir = os.path.dirname(os.path.abspath(__file__))
        shap_candidates = [
            os.path.join(base_dir, "figures", "shap_importance.png"),
            os.path.join(base_dir, "Gaitly_AI", "shap_importance.png"),
            os.path.join("figures", "shap_importance.png"),
            "shap_importance.png"
        ]
        shap_img = next((p for p in shap_candidates if os.path.exists(p)), None)

        if shap_img:
            st.subheader("🔍 تفسير الذكاء الاصطناعي (SHAP Explainability)")
            st.info("يوضح مخطط SHAP أدناه الأهمية النسبية لكل متغير بيوميكانيكي في تحديد قرار التصنيف، مما يمنح النظام شفافية كاملة (Explainable AI).")
            st.image(shap_img, caption="Global Feature Importance (SHAP TreeExplainer)", use_column_width=True)

        st.markdown("""
        <div class="disclaimer">
            <strong>⚕️ إخلاء مسؤولية سريرية:</strong> تصنيف هذا النظام أداة فحص استرشادية مبكرة (Screening Tool)
            وليست تشخيصاً طبياً نهائياً. يجب مراجعة الطبيب المختص أو أخصائي التأهيل الحركي للتقييم السريري الشامل.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👆 يرجى إتمام التحليل في التبويب الأول أولاً.")

# -----------------------------------------------------------------------------
# التبويب 4: التوصيات السريرية وتصدير تقرير PDF
# -----------------------------------------------------------------------------
with tab4:
    st.header("💡 التوصيات الإرشادية وتصدير التقرير السريري")

    if st.session_state.get("analysis_done", False):
        recs = st.session_state["recommendations"]

        st.subheader("🚩 المؤشرات المرصودة (Detected Indicators)")
        for flag in recs['Flags']:
            st.markdown(f"- {flag}")

        st.markdown("---")
        st.subheader("📋 التوجيهات والتوصيات المقترحة (Guidance)")
        for i, rec in enumerate(recs['Recommendations'], 1):
            st.markdown(f"""
            <div class="metric-card">
                <strong>توصية {i}:</strong><br/>
                {rec}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📄 تصدير التقرير السريري الرسمي (Generate PDF Report)")

        if st.button("📥 توليد تقرير PDF رسمي للطباعة", type="primary"):
            with st.spinner("جاري إنشاء وتجميع التقرير السريري بصيغة PDF..."):
                try:
                    from reportlab.lib.pagesizes import A4
                    from reportlab.lib import colors
                    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
                    from reportlab.lib.units import inch

                    pdf_filename = f"GaitlyAI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    doc = SimpleDocTemplate(pdf_filename, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
                    styles = getSampleStyleSheet()
                    story = []

                    title_style = ParagraphStyle(
                        'CustomTitle',
                        parent=styles['Heading1'],
                        fontSize=18,
                        textColor=colors.HexColor('#1e3c72'),
                        spaceAfter=10,
                        alignment=1
                    )
                    story.append(Paragraph("<b>Gaitly AI - Pediatric Gait Analysis Report</b>", title_style))
                    story.append(Paragraph("<font color='#666666'>AI & Computer Vision Markerless Screening System</font>", ParagraphStyle('Sub', parent=styles['Normal'], alignment=1, spaceAfter=20)))
                    story.append(Spacer(1, 0.15*inch))

                    # Metadata Table
                    meta_data = [
                        ["Report Date:", datetime.now().strftime('%Y-%m-%d %H:%M'), "Subject Age:", f"{st.session_state.features['Age']} years"],
                        ["Institution:", "University of Tabuk", "Methodology:", "RGB MediaPipe Pose Kinematics"]
                    ]
                    meta_table = Table(meta_data, colWidths=[1.8*inch, 2.2*inch, 1.4*inch, 2.1*inch])
                    meta_table.setStyle(TableStyle([
                        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1e3c72')),
                        ('TEXTCOLOR', (2, 0), (2, -1), colors.HexColor('#1e3c72')),
                        ('LINEBELOW', (0, -1), (-1, -1), 1, colors.HexColor('#2a5298'))
                    ]))
                    story.append(meta_table)
                    story.append(Spacer(1, 0.2*inch))

                    # Classification
                    story.append(Paragraph("<b>1. Classification Result</b>", styles['Heading2']))
                    pat_color = '#dc3545' if st.session_state.prediction == 'Atypical' else '#28a745'
                    story.append(Paragraph(
                        f"Pattern: <font color='{pat_color}'><b>{st.session_state.prediction}</b></font> | Confidence: <b>{st.session_state.confidence:.2%}</b>",
                        styles['Normal']
                    ))
                    story.append(Spacer(1, 0.15*inch))

                    # Biomechanical Features
                    story.append(Paragraph("<b>2. Biomechanical Features vs Normative Baselines:</b>", styles['Heading2']))
                    feature_data = [["Feature", "Value", "Baseline Reference (Ages 4-10)"]]
                    baselines = {
                        "Mean_KneeAngle_L": "140° - 155°",
                        "Mean_KneeAngle_R": "140° - 155°",
                        "Cadence": "90 - 120 steps/min",
                        "Symmetry_Knee": "< 10.0% (Robinson Index)"
                    }
                    for k, v in st.session_state.features.items():
                        if k in baselines:
                            feature_data.append([k.replace("_", " "), f"{v:.2f}", baselines[k]])

                    feature_table = Table(feature_data, colWidths=[2.8*inch, 1.8*inch, 2.9*inch])
                    feature_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2a5298')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc'))
                    ]))
                    story.append(feature_table)
                    story.append(Spacer(1, 0.2*inch))

                    # Confusion Matrix Image
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    cm_candidates = [
                        os.path.join(base_dir, "figures", "confusion_matrix.png"),
                        os.path.join(base_dir, "Gaitly_AI", "confusion_matrix.png"),
                        "confusion_matrix.png"
                    ]
                    cm_img = next((p for p in cm_candidates if os.path.exists(p)), None)
                    if cm_img:
                        story.append(Paragraph("<b>3. Model Performance (5-Fold Stratified CV):</b>", styles['Heading2']))
                        story.append(Image(cm_img, width=4.5*inch, height=3.0*inch))
                        story.append(Spacer(1, 0.15*inch))

                    # Clinical Guidance (clean English representation for ReportLab)
                    story.append(Paragraph("<b>4. Clinical Guidance:</b>", styles['Heading2']))
                    en_guidance = recs.get('Clinical_Guidance_EN', [])
                    if not en_guidance:
                        en_guidance = ["Routine developmental pediatric monitoring recommended."]
                    for i, g_text in enumerate(en_guidance, 1):
                        story.append(Paragraph(f"• <b>Guidance {i}:</b> {g_text}", styles['Normal']))
                        story.append(Spacer(1, 0.05*inch))

                    story.append(Spacer(1, 0.15*inch))
                    story.append(Paragraph(
                        "<b>Medical Disclaimer:</b> This report is generated as an investigational screening aid only and does not constitute a certified medical diagnosis.",
                        ParagraphStyle('Disc', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor('#721c24'))
                    ))

                    doc.build(story)

                    with open(pdf_filename, "rb") as f_pdf:
                        st.download_button(
                            label="📥 تحميل تقرير PDF الآن (Download Report)",
                            data=f_pdf,
                            file_name=pdf_filename,
                            mime="application/pdf"
                        )
                    st.success(f"✅ تم توليد التقرير بنجاح: {pdf_filename}")

                except ImportError:
                    st.error("❌ حزمة reportlab غير مثبتة. يرجى تثبيتها عبر: pip install reportlab")
                except Exception as e:
                    st.error(f"❌ حدث خطأ أثناء إنشاء التقرير: {str(e)}")
    else:
        st.info("👆 يرجى إتمام التحليل في التبويب الأول أولاً.")

# ==========================================
# 8. التذييل الأكاديمي
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1.2rem; font-size: 0.9em;'>
    <p><strong>Gaitly AI</strong> | نظام ذكي لتحليل مشية وحركة الأطفال</p>
    <p>جامعة تبوك | كلية الحاسبات وتقنية المعلومات | العام الجامعي 2026 / 2027 م</p>
    <p>إشراف: د. جوهرة الجابري</p>
</div>
""", unsafe_allow_html=True)