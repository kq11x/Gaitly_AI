"""
================================================================================
Gaitly AI - Standalone CLI Skeleton Extraction Script
أداة استخراج الهيكل العظمي وتتبع نقاط المفاصل عبر سطر الأوامر (CLI)
================================================================================
Extracts 33 skeletal landmarks using MediaPipe Pose (BlazePose) and renders
an annotated video with tracking lines, joint confidence badges, and indicators.

Academic Reference:
- Chapter 3: Section 3.2.1 (FR1), Section 3.3.1 (Algorithm 3.1)
- MediaPipe Pose: On-device Real-time Body Pose Tracking (Bazrev et al., CVPR 2020)
================================================================================
"""

import cv2
import mediapipe as mp
import os
import sys
import argparse
from pathlib import Path

# Safe UTF-8 encoding on Windows terminal
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass


def find_default_video(project_root: Path) -> Path:
    """Discovers available video samples dynamically across raw_videos and viduo directories."""
    candidates = [
        project_root / "data" / "raw_videos" / "sample_child_gait_01.mp4",
        project_root / "data" / "raw_videos" / "sample_child_gait_02.mp4",
        project_root / "data" / "raw_videos" / "pediatric_gait_subject_01_sagittal.mp4",
        project_root / "viduo" / "WhatsApp Video 2026-10-17 at 1.13.06 AM.mp4",
        project_root / "sample_child_gait_01.mp4"
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def parse_args():
    project_root = Path(__file__).resolve().parent.parent
    default_input = find_default_video(project_root)
    default_output = project_root / "data" / "outputs" / "output_skeleton.mp4"

    parser = argparse.ArgumentParser(
        description="Gaitly AI - Video Skeleton Landmark Extraction Tool (MediaPipe Pose)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(default_input),
        help=f"Path to input video (default: {default_input})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(default_output),
        help=f"Path to output annotated video (default: {default_output})"
    )
    parser.add_argument(
        "--complexity", "-c",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="MediaPipe model complexity: 0=Lite, 1=Full (Default), 2=Heavy"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_video_path = args.input
    output_video_path = args.output

    project_root = Path(__file__).resolve().parent.parent

    # Dynamic fallback search if requested video doesn't exist
    if not os.path.exists(input_video_path):
        fallback = find_default_video(project_root)
        if fallback.exists():
            print(f"⚠️ لم يتم العثور على '{input_video_path}'، سيتم استخدام الفيديو المتاح: '{fallback}'")
            input_video_path = str(fallback)
        else:
            print(f"❌ خطأ: لم يتم العثور على الفيديو '{input_video_path}'")
            sys.exit(1)

    print("=" * 70)
    print("🚶‍♂️ [Gaitly AI] Video Skeleton Landmark Extraction Tool")
    print(f"📂 Input Video  : {input_video_path}")
    print(f"💾 Output Video : {output_video_path}")
    print(f"⚙️ Complexity   : {args.complexity} (0=Lite, 1=Full, 2=Heavy)")
    print("=" * 70)

    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)

    # Initialize MediaPipe Pose pipeline
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=args.complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"❌ خطأ في فتح ملف الفيديو: {input_video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, int(fps), (width, height))

    print(f"✅ بدأت المعالجة: {total_frames} إطار بمعدل {fps:.1f} إطار/ثانية...")

    frame_count = 0
    detected_count = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % 30 == 0 or frame_count == total_frames:
            pct = (frame_count / total_frames * 100) if total_frames > 0 else 0
            print(f"⏳ جاري المعالجة... الإطار {frame_count} من {total_frames} ({pct:.1f}%)")

        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            detected_count += 1
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Highlight Right Knee (Landmark #26)
            right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
            rk_x_px = int(right_knee.x * width)
            rk_y_px = int(right_knee.y * height)
            visibility = right_knee.visibility

            cv2.circle(image, (rk_x_px, rk_y_px), 7, (0, 255, 0), -1)
            cv2.circle(image, (rk_x_px, rk_y_px), 9, (255, 255, 255), 2)
            cv2.putText(
                image,
                f"R-Knee Conf: {visibility:.2f}",
                (rk_x_px + 12, rk_y_px - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        out.write(image)

    cap.release()
    out.release()
    pose.close()

    det_rate = (detected_count / frame_count * 100) if frame_count > 0 else 0
    print(f"\n🎉 تم الانتهاء بنجاح! نسبة اكتشاف الوضعية: {det_rate:.1f}%")
    print(f"💾 تم حفظ الفيديو المعالج في:\n{output_video_path}\n")


if __name__ == "__main__":
    main()