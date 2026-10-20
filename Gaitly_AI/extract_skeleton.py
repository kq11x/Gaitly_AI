import cv2
import mediapipe as mp
import os

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
# 2. إعدادات الفيديو (الإدخال والإخراج)
# ==========================================
input_video_path = r"C:\Users\hacker\Desktop\machine_learnig\Gaitly_AI\viduo\BHAI.mp4"  # ضع اسم الفيديو التجريبي هنا
output_video_path = "output_skeleton.mp4"

# التحقق من وجود الفيديو
if not os.path.exists(input_video_path):
    print(f"❌ خطأ: لم يتم العثور على الفيديو '{input_video_path}'")
    exit()

# فتح الفيديو
cap = cv2.VideoCapture(input_video_path)

# استخراج خصائص الفيديو الأصلي للحفاظ على الجودة عند الحفظ
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# تعريف تنسيق الحفظ (MP4V متوافق مع معظم الأنظمة)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"✅ بدأ المعالجة: {total_frames} إطار بمعدل {fps} إطار/ثانية...")

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
    if frame_count % 30 == 0:  # طباعة تقدم كل 30 إطار (تقريباً كل ثانية)
        print(f"⏳ جاري المعالجة... الإطار {frame_count} من {total_frames}")

    # تحسين الأداء: جعل الصورة قابلة للكتابة
    image.flags.writeable = False
    
    # MediaPipe تتوقع صوراً بصيغة RGB، بينما OpenCV يقرأ بصيغة BGR
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # معالجة الإطار واستخراج المفاصل
    results = pose.process(image_rgb)

    # إعادة الصورة لصيغة BGR للرسم والحفظ
    image.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # ==========================================
    # 4. الرسم واستخراج البيانات
    # ==========================================
    if results.pose_landmarks:
        # أ) رسم الهيكل العظمي والاتصالات
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # ب) إثبات استخراج البيانات: عرض إحداثيات وثقة مفصل محوري (مثلاً: الركبة اليمنى - Landmark 26)
        # ملاحظة: MediaPose يستخدم 33 نقطة. 25=Left Knee, 26=Right Knee
        right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        
        # تحويل الإحداثيات النسبية (0.0 إلى 1.0) إلى بكسل
        rk_x_px = int(right_knee.x * width)
        rk_y_px = int(right_knee.y * height)
        visibility = right_knee.visibility # درجة الثقة (0.0 إلى 1.0)

        # رسم دائرة صغيرة على الركبة اليمنى ونص يوضح درجة الثقة
        cv2.circle(image, (rk_x_px, rk_y_px), 6, (0, 255, 0), -1)
        cv2.putText(image, f"R-Knee Conf: {visibility:.2f}", 
                    (rk_x_px + 10, rk_y_px - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # كتابة الإطار المعالج إلى فيديو الإخراج
    out.write(image)

# ==========================================
# 5. تنظيف الموارد
# ==========================================
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"🎉 تم الانتهاء بنجاح! تم حفظ الفيديو المعالج باسم: {output_video_path}")