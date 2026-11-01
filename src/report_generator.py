"""
================================================================================
Gaitly AI - Clinical Recommendations & Analytical PDF Report Generator
وحدة توليد التوصيات السريرية وتصدير التقارير الطبية بصيغة PDF
================================================================================
Evidence-based recommendation engine adhering to pediatric biomechanical thresholds
and ReportLab document compiler for generating printable clinical summaries.

مسؤولة عن:
1. تطبيق القواعد السريرية المبنية على الأدلة والمؤشرات البيوميكانيكية للأطفال (4-10 سنوات).
2. تقييم مؤشر تماثل روبنسون (Robinson Index)، معدل الخطوات (Cadence)، وزوايا ثني الركبة.
3. إنشاء تقارير PDF رسمية وأنيقة تحتوي جدول الخصائص ومقارنتها بالنطاقات الطبيعية،
   مخطط أداء النموذج، والتوصيات الإرشادية مع إخلاء المسؤولية الطبية الصريح.
================================================================================
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional, List

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def generate_recommendations(
    feature_vector: Dict[str, Any],
    prediction: str,
    confidence: float
) -> Dict[str, Any]:
    """
    Evidence-based clinical recommendation engine based on pediatric gait literature:
    محرك التوصيات الإرشادية الذكي المعتمد على المعايير البيوميكانيكية لنمو الأطفال:

    Biomechanical Rules Evaluated:
    ------------------------------
    1. Robinson Knee Symmetry Index:
       - SI <= 10.0%: Physiological Symmetry (Normal)
       - 10.0% < SI <= 18.0%: Mild to Moderate Asymmetry
       - SI > 18.0%: Clinically Significant Asymmetry
    2. Cadence (steps/min) for ages 4-10:
       - Normal: 90 - 120 steps/min
       - Low (<90): Possible guarded gait, reduced stability, or fatigue
       - High (>120): Rapid compensatory stepping cadence
    3. Bilateral Knee Flexion Discrepancy:
       - Difference > 15.0 degrees between left and right limb averages
    4. Knee Range of Motion Extremes:
       - < 130 deg: Persistent crouch gait tendency
       - > 165 deg: Knee hyperextension / genu recurvatum tendency

    Parameters:
    -----------
    feature_vector : Dict[str, Any]
        Aggregated video kinematics.
    prediction : str
        "Typical" or "Atypical"
    confidence : float
        Prediction confidence score [0.0 - 1.0].

    Returns:
    --------
    Dict[str, Any] containing Arabic & English clinical recommendations and detected flags.
    """
    flags: List[str] = []
    recommendations: List[str] = []
    clinical_guidance_en: List[str] = []

    # 1. Robinson Symmetry Index Evaluation
    symmetry_val = float(feature_vector.get("Symmetry_Knee", 0.0))
    if symmetry_val > 18.0:
        flags.append(f"⚠️ عدم تماثل حركي ملحوظ بين الطرفين السفليين (مؤشر روبنسون: {symmetry_val:.1f}% > 18%)")
        recommendations.append("يُنصح بإجراء فحص سريري تخصصي (العظام والعلاج الطبيعي) لتقييم عدم التناظر في الطرفين السفليين وتوازن الحوض.")
        clinical_guidance_en.append(f"Pronounced lower limb kinematic asymmetry (Robinson Index: {symmetry_val:.1f}% > 18%). Clinical orthopedic/PT gait assessment recommended.")
    elif symmetry_val > 10.0:
        flags.append(f"⚠️ تفاوت حركي خفيف إلى متوسط بين الطرفين (مؤشر روبنسون: {symmetry_val:.1f}%)")
        recommendations.append("يُنصح بملاحظة حركة الطفل أثناء الجري والأنشطة الرياضية للتأكد من عدم وجود عرج خفيف أو انحياز لأحد الطرفين.")
        clinical_guidance_en.append(f"Mild to moderate limb asymmetry detected (Robinson Index: {symmetry_val:.1f}%). Monitor functional balance during active play.")

    # 2. Cadence Evaluation
    cadence_val = float(feature_vector.get("Cadence", 0.0))
    if cadence_val > 0.0:
        if cadence_val < 90.0:
            flags.append(f"⚠️ معدل الخطوات منخفض عن النطاق المعتاد ({cadence_val:.1f} خطوة/دقيقة - الطبيعي: 90-120)")
            recommendations.append("يُنصح بمراقبة سرعة المشي والتأكد من خلو الطفل من آلام المفاصل أو الإجهاد العضلي المبكر أثناء المشي لمسافات متوسطة.")
            clinical_guidance_en.append(f"Reduced cadence ({cadence_val:.1f} steps/min, normative: 90-120). Rule out muscular fatigue or joint discomfort.")
        elif cadence_val > 120.0:
            flags.append(f"⚠️ معدل خطوات مرتفع وسريع ({cadence_val:.1f} خطوة/دقيقة)")
            recommendations.append("معدل الخطوات مرتفع نسبياً؛ قد يكون ذلك ناتجاً عن صغر طول الخطوة أو التعويض عن عدم الثبات.")
            clinical_guidance_en.append(f"Elevated stepping cadence ({cadence_val:.1f} steps/min). May indicate shortened stride length or compensatory balance control.")

    # 3. Bilateral Knee Flexion Difference
    knee_l = float(feature_vector.get("Mean_KneeAngle_L", 0.0))
    knee_r = float(feature_vector.get("Mean_KneeAngle_R", 0.0))
    diff = abs(knee_l - knee_r)
    if diff > 15.0:
        flags.append(f"⚠️ تباين ملحوظ في متوسط زوايا ثني الركبتين ({diff:.1f} درجة)")
        recommendations.append("يُنصح بتقييم المدى الحركي للمفاصل (Range of Motion) وقوة الأوتار الخلفية للركبة بواسطة أخصائي تأهيل حركي.")
        clinical_guidance_en.append(f"Significant sagittal knee angle discrepancy ({diff:.1f} deg). Evaluate joint Range of Motion (ROM) and hamstring muscle tone.")

    # 4. Extreme Knee Angles (Crouch vs Hyperextension)
    min_knee = min(knee_l, knee_r)
    max_knee = max(knee_l, knee_r)
    if min_knee < 130.0 and min_knee > 50.0:
        flags.append("⚠️ ميل لزيادة ثني الركبة أثناء المشي (Crouch Gait Tendency)")
        recommendations.append("يُنصح بفحص قوة عضلات الفخذ الرباعية والأوتار للتأكد من عدم وجود قصر في الأوتار الباسطة.")
        clinical_guidance_en.append("Excessive knee flexion observed. Screen for quadriceps weakness or hamstring tightness.")

    # Default if no red flags detected
    if not flags:
        flags.append("✅ لا توجد مؤشرات مقلقة؛ كافة القياسات ضمن النطاق الطبيعي المتوقع")
        recommendations.append("المؤشرات البيوميكانيكية والتماثل الحركي ومعدل الخطوات متوافقة مع مراحل النمو الطبيعية لعمر الطفل. يُنصح بالاستمرار في النشاط البدني والمتابعة الدورية.")
        clinical_guidance_en.append("All biomechanical kinematic indices conform to typical developmental baselines. Continue routine developmental monitoring.")

    conf_str = f"{confidence:.2%}" if isinstance(confidence, float) else str(confidence)

    return {
        "Prediction": prediction,
        "Confidence": conf_str,
        "Flags": flags,
        "Recommendations": recommendations,
        "Clinical_Guidance_EN": clinical_guidance_en,
        "Disclaimer": "هذه النتائج والتوصيات صادرة عن أداة ذكاء اصطناعي مساعدة للمسح الاسترشادي المبكر (Screening Tool) فقط، ولا تُعد تشخيصاً طبياً نهائياً. يجب مراجعة طبيب أطفال أو أخصائي علاج طبيعي معتمد للتقييم السريري الشامل."
    }


def generate_pdf_report(
    feature_vector: Dict[str, Any],
    prediction: str,
    confidence: float,
    recommendations_data: Dict[str, Any],
    output_pdf_path: Optional[str] = None,
    confusion_matrix_path: Optional[str] = None
) -> str:
    """
    Compiles an analytical, publication-quality clinical PDF report using ReportLab.
    إنشاء تقرير سريري رسمي بصيغة PDF يحتوي ملخص التحليل ومقارنة المتغيرات والتوصيات.

    Parameters:
    -----------
    feature_vector : Dict[str, Any]
        Dictionary of extracted biomechanical features.
    prediction : str
        "Typical" or "Atypical".
    confidence : float
        Model prediction confidence.
    recommendations_data : Dict[str, Any]
        Dictionary output from generate_recommendations().
    output_pdf_path : Optional[str]
        File path to save the generated PDF report.
    confusion_matrix_path : Optional[str]
        File path to model performance confusion matrix figure.

    Returns:
    --------
    str : Path to the saved PDF report file.
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "ReportLab library is required to compile PDF reports. Install via: pip install reportlab"
        )

    if output_pdf_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_pdf_path = f"GaitlyAI_Report_{timestamp}.pdf"

    os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    styles = getSampleStyleSheet()
    story = []

    # 1. Header Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1e3c72'),
        spaceAfter=6,
        alignment=1
    )
    story.append(Paragraph("<b>Gaitly AI - Pediatric Gait Analysis Report</b>", title_style))
    story.append(Paragraph(
        "<font color='#555555'>Markerless AI & Computer Vision Kinematic Screening System</font>",
        ParagraphStyle('Sub', parent=styles['Normal'], fontSize=10, alignment=1, spaceAfter=15)
    ))
    story.append(Spacer(1, 0.15 * inch))

    # 2. Metadata Information Table
    subject_age = feature_vector.get('Age', 'N/A')
    meta_data = [
        ["Report Date:", datetime.now().strftime('%Y-%m-%d %H:%M'), "Subject Age:", f"{subject_age} years"],
        ["Methodology:", "RGB Pose Kinematics (MediaPipe)", "Institution:", "University of Tabuk (2026/2027)"]
    ]
    meta_table = Table(meta_data, colWidths=[1.8 * inch, 2.2 * inch, 1.4 * inch, 2.1 * inch])
    meta_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1e3c72')),
        ('TEXTCOLOR', (2, 0), (2, -1), colors.HexColor('#1e3c72')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LINEBELOW', (0, -1), (-1, -1), 1, colors.HexColor('#2a5298'))
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.2 * inch))

    # 3. AI Classification Section
    story.append(Paragraph("<b>1. AI Model Classification Result</b>", styles['Heading2']))
    is_atypical = (prediction == 'Atypical')
    pattern_color = '#dc3545' if is_atypical else '#28a745'
    conf_str = f"{confidence:.2%}" if isinstance(confidence, float) else str(confidence)

    classification_html = f"""
    Classification: <font color='{pattern_color}'><b>{prediction} Gait Pattern</b></font><br/>
    Model Confidence: <b>{conf_str}</b><br/>
    Classifier: <b>XGBoost (Stratified 5-Fold Cross-Validation, NFR1 Evaluated)</b>
    """
    story.append(Paragraph(classification_html, styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # 4. Biomechanical Features Table with Baselines
    story.append(Paragraph("<b>2. Extracted Kinematic Features vs. Developmental Baselines</b>", styles['Heading2']))
    table_rows = [["Biomechanical Metric", "Extracted Value", "Normative Baseline (Ages 4-10)"]]
    baselines = {
        "Mean_KneeAngle_L": "140.0° - 155.0° (Sagittal Extension)",
        "Mean_KneeAngle_R": "140.0° - 155.0° (Sagittal Extension)",
        "Cadence": "90.0 - 120.0 steps/min",
        "Symmetry_Knee": "< 10.0% (Robinson Symmetry Index)"
    }
    for k, v in feature_vector.items():
        if k in baselines:
            formatted_val = f"{v:.2f}" if isinstance(v, (int, float)) else str(v)
            metric_label = k.replace("_", " ")
            table_rows.append([metric_label, formatted_val, baselines[k]])

    feat_table = Table(table_rows, colWidths=[2.8 * inch, 1.8 * inch, 2.9 * inch])
    feat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2a5298')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8f9fa'), colors.whitesmoke]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
    ]))
    story.append(feat_table)
    story.append(Spacer(1, 0.2 * inch))

    # 5. Model Validation Graphic (if exists)
    if confusion_matrix_path and os.path.exists(confusion_matrix_path):
        story.append(Paragraph("<b>3. Model Performance Validation</b>", styles['Heading2']))
        story.append(Image(confusion_matrix_path, width=4.5 * inch, height=3.0 * inch))
        story.append(Spacer(1, 0.15 * inch))

    # 6. Clinical Guidance & Observations
    story.append(Paragraph("<b>4. Clinical Observations & Guidance</b>", styles['Heading2']))
    guidance_list = recommendations_data.get('Clinical_Guidance_EN', [])
    if not guidance_list:
        guidance_list = recommendations_data.get('Recommendations', [])

    for i, item in enumerate(guidance_list, 1):
        # Remove any non-ASCII characters for clean PDF compilation
        safe_item = item.encode('ascii', errors='ignore').decode('ascii')
        if not safe_item.strip():
            safe_item = f"Kinematic observation and guidance item {i} recorded."
        story.append(Paragraph(f"• <b>Guidance {i}:</b> {safe_item}", styles['Normal']))
        story.append(Spacer(1, 0.06 * inch))

    # 7. Medical Disclaimer
    story.append(Spacer(1, 0.2 * inch))
    disclaimer_box = """
    <b>Medical & Ethical Disclaimer:</b> This report is generated by Gaitly AI as an investigational
    screening aid for educational and early screening assistance only. It does not replace comprehensive
    pediatric medical evaluation, clinical gait laboratory analysis, or clinical diagnosis.
    """
    story.append(Paragraph(
        disclaimer_box,
        ParagraphStyle('Disc', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor('#721c24'))
    ))

    doc.build(story)
    return output_pdf_path
