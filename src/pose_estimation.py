"""
================================================================================
Gaitly AI - Pose Estimation Module | وحدة تقدير الوضعية واستخراج الهيكل العظمي
================================================================================
Responsible for detecting 33 full-body skeletal landmarks from pediatric walking
videos using Google MediaPipe Pose (BlazePose topology).

مسؤولة عن تتبع واستخراج 33 نقطة هيكلية عظمية ثلاثية الأبعاد للأطفال أثناء المشي
باستخدام نموذج MediaPipe Pose، مع دعم تتبع درجات الثقة (Visibility) ورسم الهيكل العظمي.

Reference:
- MediaPipe Pose: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
- BlazePose: On-device Real-time Body Pose Tracking (Bazrev et al., CVPR 2020)
================================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from typing import Optional, Tuple, Dict, Any, List


def get_pose_estimator(
    static_image_mode: bool = False,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5
) -> mp.solutions.pose.Pose:
    """
    Initializes and returns a configured MediaPipe Pose estimator instance.
    تهيئة وإرجاع كائن نموذج MediaPipe Pose بالإعدادات المناسبة للفيديو.

    Parameters:
    -----------
    static_image_mode : bool
        False for continuous video streaming (enables temporal tracking),
        True for independent static frames.
    model_complexity : int
        0 = Lite (fastest, lowest resource usage),
        1 = Full (optimal balance of accuracy and latency, default),
        2 = Heavy (highest landmark precision, slower).
    min_detection_confidence : float
        Minimum confidence threshold [0.0, 1.0] to consider person detection successful.
    min_tracking_confidence : float
        Minimum confidence threshold [0.0, 1.0] to maintain tracked landmarks across frames.

    Returns:
    --------
    mp.solutions.pose.Pose : Initialized MediaPipe Pose pipeline.
    """
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )


def extract_landmarks_from_frame(frame: np.ndarray, pose_model: Any) -> Any:
    """
    Converts a BGR video frame to RGB and processes it through MediaPipe Pose.
    تحويل الإطار من صيغة BGR إلى RGB وتمريره عبر نموذج MediaPipe لاستخراج النقاط الهيكلية.

    Parameters:
    -----------
    frame : np.ndarray
        Raw OpenCV image frame in BGR format.
    pose_model : mp.solutions.pose.Pose
        Active MediaPipe Pose pipeline instance.

    Returns:
    --------
    landmarks : NormalizedLandmarkList or None if no human pose detected in the frame.
    """
    if frame is None or frame.size == 0:
        return None

    # MediaPipe requires RGB color space
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(rgb_frame)
    return results.pose_landmarks


def get_landmark_coords(
    landmarks: Any,
    landmark_id: int,
    visibility_threshold: float = 0.5
) -> Optional[Tuple[float, float]]:
    """
    Extracts normalized (x, y) 2D coordinates for a specific MediaPipe landmark index.
    استخراج الإحداثيات ثنائية الأبعاد (x, y) لنقطة هيكلية محددة بشرط تجاوز عتبة الظهور (Visibility).

    Parameters:
    -----------
    landmarks : NormalizedLandmarkList
        Pose landmarks object returned by MediaPipe.
    landmark_id : int
        MediaPipe landmark ID index (0 to 32).
        Key lower-body landmarks:
          - 23: Left Hip | 24: Right Hip
          - 25: Left Knee | 26: Right Knee
          - 27: Left Ankle | 28: Right Ankle
          - 29: Left Heel | 30: Right Heel
          - 31: Left Foot Index | 32: Right Foot Index
    visibility_threshold : float
        Minimum visibility score [0.0 - 1.0]. Landmarks below this score are treated as occluded.

    Returns:
    --------
    Optional[Tuple[float, float]] : (x, y) normalized coordinates in range [0.0, 1.0], or None.
    """
    if landmarks is None or landmark_id >= len(landmarks.landmark):
        return None

    lm = landmarks.landmark[landmark_id]
    if getattr(lm, "visibility", 1.0) >= visibility_threshold:
        return (float(lm.x), float(lm.y))
    return None


def get_landmark_coords_3d(
    landmarks: Any,
    landmark_id: int,
    visibility_threshold: float = 0.5
) -> Optional[Tuple[float, float, float]]:
    """
    Extracts normalized 3D (x, y, z) coordinates for advanced kinematic depth modeling.
    استخراج الإحداثيات ثلاثية الأبعاد (x, y, z) للنقطة الهيكلية المحددة مع عمق z النسبي.
    """
    if landmarks is None or landmark_id >= len(landmarks.landmark):
        return None

    lm = landmarks.landmark[landmark_id]
    if getattr(lm, "visibility", 1.0) >= visibility_threshold:
        return (float(lm.x), float(lm.y), float(lm.z))
    return None


def annotate_frame_with_skeleton(
    frame: np.ndarray,
    landmarks: Any,
    highlight_joint: Optional[int] = 26  # Default: Right Knee
) -> np.ndarray:
    """
    Draws the full 33-point skeletal connections and an optional highlighted joint on the frame.
    رسم شبكة الاتصالات الهيكلية العظمية الـ 33 ونقاط المفاصل على الإطار مع إبراز مفصل الركبة.

    Parameters:
    -----------
    frame : np.ndarray
        Original video frame.
    landmarks : NormalizedLandmarkList
        Detected landmarks.
    highlight_joint : Optional[int]
        Specific landmark index to highlight with a circular indicator and confidence readout.

    Returns:
    --------
    np.ndarray : Annotated frame with visual skeleton overlays.
    """
    annotated = frame.copy()
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    if landmarks:
        # Draw full pose connections and keypoints
        mp_drawing.draw_landmarks(
            annotated,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # Highlight key anatomical joint (e.g., Right Knee #26 for gait analysis)
        if highlight_joint is not None and highlight_joint < len(landmarks.landmark):
            joint = landmarks.landmark[highlight_joint]
            h, w = annotated.shape[:2]
            jx, jy = int(joint.x * w), int(joint.y * h)
            conf = getattr(joint, "visibility", 1.0)

            # Draw glowing marker circle
            cv2.circle(annotated, (jx, jy), 7, (0, 255, 0), -1)
            cv2.circle(annotated, (jx, jy), 9, (255, 255, 255), 2)
            cv2.putText(
                annotated,
                f"Knee Conf: {conf:.2f}",
                (jx + 12, jy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    return annotated


def process_video_skeleton(
    input_video_path: str,
    output_video_path: Optional[str] = None,
    model_complexity: int = 1,
    log_interval: int = 30
) -> Dict[str, Any]:
    """
    Batch processes an entire video to extract skeletal landmarks and optionally render
    an annotated output video file.
    معالجة مقطع فيديو كامل، استخراج النقاط الهيكلية، وحفظ فيديو المخرجات المرسوم عليه الهيكل.

    Parameters:
    -----------
    input_video_path : str
        Path to the source RGB walking video (MP4/MOV/AVI).
    output_video_path : Optional[str]
        Destination path to save the annotated video (e.g. data/outputs/tracked.mp4).
    model_complexity : int
        0 (Lite), 1 (Full, Recommended), 2 (Heavy).
    log_interval : int
        Frames interval for logging progress messages.

    Returns:
    --------
    Dict[str, Any] containing metadata: total_frames, detected_frames, detection_rate_pct, fps.
    """
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video stream: {input_video_path}")

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(raw_fps) if (raw_fps is not None and raw_fps > 0 and not np.isnan(raw_fps)) else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = None
    if output_video_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, int(fps), (width, height))

    pose = get_pose_estimator(model_complexity=model_complexity)
    frame_idx = 0
    detected_frames = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        landmarks = extract_landmarks_from_frame(frame, pose)
        if landmarks:
            detected_frames += 1

        if out is not None:
            annotated = annotate_frame_with_skeleton(frame, landmarks)
            out.write(annotated)

        frame_idx += 1
        if frame_idx % log_interval == 0:
            print(f"⏳ Processing frame {frame_idx}/{total_frames}...")

    cap.release()
    if out is not None:
        out.release()
    pose.close()

    detection_rate = (detected_frames / frame_idx * 100.0) if frame_idx > 0 else 0.0

    return {
        "total_frames": frame_idx,
        "detected_frames": detected_frames,
        "detection_rate_pct": round(detection_rate, 2),
        "fps": fps,
        "dimensions": (width, height),
        "output_video": output_video_path
    }
