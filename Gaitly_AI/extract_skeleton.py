"""
================================================================================
Gaitly AI - Video Skeleton Landmark Extraction Tool (MediaPipe Pose)
أداة استخراج الهيكل العظمي وتتبع مفاصل المشية للأطفال
================================================================================
"""

import cv2
import mediapipe as mp
import os
from pathlib import Path

# ==========================================
# 1. إعدادات MediaPipe والرسم
# ==========================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# تهيئة نموذج Pose (مضبوط على الفيديو الحقيقي وليس الصور الثابتة)
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,       # 1 للتوازن بين السرعة والدقة (مناسب للأجهزة العادية)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================
# 2. إعدادات الفيديو (مسارات نسبية مرنة)
# ==========================================
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

possible_inputs = [
    parent_dir / "data" / "raw_videos" / "sample_child_gait_02.mp4",
    parent_dir / "data" / "raw_videos" / "sample_child_gait_01.mp4",
    parent_dir / "viduo" / "WhatsApp Video 2026-10-17 at 1.13.06 AM.mp4",
    current_dir / "sample_child_gait_02.mp4"
]

input_video_path = next((str(p) for p in possible_inputs if p.exists()), str(possible_inputs[0]))
output_video_path = str(parent_dir / "data" / "outputs" / "output_skeleton.mp4")

# التحقق من وجود الفيديو
if not os.path.exists(input_video_path):
    print(f"❌ خطأ: لم يتم العثور على الفيديو '{input_video_path}'")
    exit(1)

# فتح الفيديو
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)

# تعريف تنسيق الحفظ (MP4V متوافق مع معظم الأنظمة)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"✅ بدأ المعالجة: {total_frames} إطار بمعدل {fps} إطار/ثانية...")
print(f"📂 الفيديو المدخل: {input_video_path}")
print(f"💾 مسار الحفظ: {output_video_path}")

# ==========================================
# 3. حلقة المعالجة الرئيسية (Frame-by-Frame)
# ==========================================
frame_count = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("⚠️ انتهى الفيديو أو حدث خطأ في القراءة.")
        break

    frame_count += 1
    if frame_count % 30 == 0 or frame_count == total_frames:
        pct = (frame_count / total_frames * 100) if total_frames > 0 else 0
        print(f"⏳ جاري المعالجة... الإطار {frame_count} من {total_frames} ({pct:.1f}%)")

    # تحسين الأداء: جعل الصورة غير قابلة للتعديل أثناء المعالجة
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    image.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # التحقق من اكتشاف نقاط الهيكل العظمي
    if results.pose_landmarks:
        # رسم الهيكل العظمي بالكامل
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # تحديد وإبراز نقطة الركبة اليمنى (المفصل رقم 26)
        right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        rk_x_px = int(right_knee.x * width)
        rk_y_px = int(right_knee.y * height)
        visibility = right_knee.visibility

        # رسم دائرة خضراء وكتابة نسبة الثقة فوق المفصل
        cv2.circle(image, (rk_x_px, rk_y_px), 6, (0, 255, 0), -1)
        cv2.putText(
            image,
            f"R-Knee Conf: {visibility:.2f}",
            (rk_x_px + 10, rk_y_px - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    out.write(image)

# ==========================================
# 4. إغلاق وتحرير الموارد
# ==========================================
cap.release()
out.release()
pose.close()
cv2.destroyAllWindows()

print(f"🎉 تم الانتهاء بنجاح! تم حفظ الفيديو المعالج في:\n{output_video_path}")