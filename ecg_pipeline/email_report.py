"""
ecg_pipeline.email_report
============================
يبني ويُرسل تقرير تصنيف تسجيل واحد عبر البريد الإلكتروني: شكل بياني لكل
قطب (النبضة الفعلية فوق المدى الطبيعي المظلَّل + الانحراف بعد Stemming)،
جدول تفصيلي (احتمالات كل فئة لكل قطب)، وجدول مبسَّط (نسبة تصنيف إجمالية
لكل فئة عبر كل الأقطاب معاً) -- مصمَّم ليكون مادة مراجعة كافية لطبيب/فني
بدراسة ميدانية للتحقق من دقة التصنيف.

⚠️ مستقل تماماً عن إطار التطبيق (Flask/FastAPI/Streamlit/...) -- يُستدعى
بدالة واحدة `generate_and_send_report(...)` من أي مكان بالتطبيق الحالي
فور اكتمال التصنيف. لا يحتاج معرفة كيف يعمل باقي التطبيق.

⚠️ يستخدم SMTP عادي (smtplib، مكتبة قياسية بلا تبعيات خارجية) بما إنه
لا توجد آلية إرسال بريد مفعَّلة حالياً. يحتاج:
  - حساب بريد مع "كلمة مرور تطبيق" (App Password) -- ليس كلمة المرور
    العادية، خصوصاً لو Gmail (يُفعَّل من إعدادات الحساب Google → الأمان
    → التحقق بخطوتين → كلمات مرور التطبيقات).
  - أو خادم SMTP خاص بالشركة/الجهة إذا متوفر.
  لا تُخزَّن بيانات الاعتماد داخل الكود إطلاقاً -- مرَّرها متغيرات بيئة
  (راجع SMTPConfig.from_env أدناه).
"""

from __future__ import annotations
import io
import os
import smtplib
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText

import numpy as np
import matplotlib
matplotlib.use("Agg")  # لا حاجة لواجهة رسومية على سيرفر السحابة
import matplotlib.pyplot as plt


# ============================================================
# 1) إعدادات SMTP (من متغيرات البيئة -- لا كلمات مرور بالكود)
# ============================================================
@dataclass
class SMTPConfig:
    host: str
    port: int
    username: str
    password: str
    sender_email: str
    use_tls: bool = True

    @classmethod
    def from_env(cls) -> "SMTPConfig":
        """
        يقرأ الإعدادات من متغيرات البيئة:
          ECG_SMTP_HOST     (مثال: smtp.gmail.com)
          ECG_SMTP_PORT     (مثال: 587)
          ECG_SMTP_USERNAME (عنوان البريد المُرسِل)
          ECG_SMTP_PASSWORD (كلمة مرور التطبيق -- ليست كلمة المرور العادية)
          ECG_SMTP_SENDER   (اختياري -- يساوي USERNAME افتراضياً)
        """
        host = os.environ["ECG_SMTP_HOST"]
        port = int(os.environ.get("ECG_SMTP_PORT", "587"))
        username = os.environ["ECG_SMTP_USERNAME"]
        password = os.environ["ECG_SMTP_PASSWORD"]
        sender = os.environ.get("ECG_SMTP_SENDER", username)
        return cls(host=host, port=port, username=username, password=password, sender_email=sender)


# ============================================================
# 2) شكل بياني لكل قطب (نبضة المريض الفعلية فوق المدى الطبيعي)
# ============================================================
def render_lead_figure(lead: str, beat: np.ndarray, ref_min: np.ndarray, ref_max: np.ndarray,
                        cutoff: int, pre: int, predicted_class: str) -> bytes:
    """
    يرسم نبضة مريض واحد (بعد كل المعالجات: تمرير عالٍ، تصحيح محلي) فوق
    المدى الطبيعي المظلَّل لنفس القطب، مع تظليل رمادي لأي جزء مقنَّع
    (تجاوز نقطة القطع الديناميكية). يرجع الصورة كـ bytes (PNG) جاهزة
    للتضمين المباشر بالإيميل (inline، لا كمرفق منفصل).

    ⚠️ نصوص الشكل بالإنجليزية عمداً: matplotlib يعرض النص العربي معكوساً
    ومفكّكاً بلا مكتبات تشكيل إضافية (arabic_reshaper + python-bidi) غير
    مضمونة التوفّر بكل بيئة سحابية. النص العربي بجسم الإيميل والجداول
    (HTML) يُعرَض صحيحاً بشكل طبيعي عبر المتصفح/برنامج البريد، فلا داعي
    لأي حل هنا سوى إبقاء نص الشكل نفسه إنجليزياً.
    """
    length = len(beat)
    t_axis = np.arange(-pre, length - pre)

    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.fill_between(t_axis, ref_min, ref_max, color="tab:blue", alpha=0.2, label="Normal range")
    ax.plot(t_axis[:cutoff], beat[:cutoff], color="tab:orange", linewidth=1.3, label="Patient beat")
    if cutoff < length:
        ax.axvspan(t_axis[cutoff], t_axis[-1], color="gray", alpha=0.15)
    ax.axvline(0, color="red", linestyle="--", linewidth=0.7)
    ax.set_title(f"{lead} — predicted: {predicted_class}", fontsize=11)
    ax.set_xlabel("Time relative to R-peak (samples)", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=8)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ============================================================
# 3) الجدولان (تفصيلي + مبسَّط)
# ============================================================
def build_detailed_table_html(lead_results: dict[str, dict]) -> str:
    """
    lead_results: {lead name: {"predicted": predicted class, "probs": {class: probability}}}
    Builds an HTML table: one row per lead, one column per possible class + a "Predicted" column.
    """
    all_classes = sorted({c for r in lead_results.values() for c in r["probs"]})
    header = "".join(f"<th style='padding:6px 10px;border:1px solid #ccc'>{c}</th>" for c in all_classes)
    rows = ""
    for lead, r in lead_results.items():
        cells = "".join(
            f"<td style='padding:6px 10px;border:1px solid #ccc;text-align:center'>"
            f"{r['probs'].get(c, 0.0) * 100:.1f}%</td>"
            for c in all_classes
        )
        predicted_style = "font-weight:bold;color:#b00020"
        rows += (
            f"<tr><td style='padding:6px 10px;border:1px solid #ccc'>{lead}</td>"
            f"<td style='padding:6px 10px;border:1px solid #ccc;{predicted_style}'>{r['predicted']}</td>"
            f"{cells}</tr>"
        )
    return (
        "<table style='border-collapse:collapse;font-family:Arial,sans-serif;font-size:13px'>"
        f"<tr><th style='padding:6px 10px;border:1px solid #ccc'>Lead</th>"
        f"<th style='padding:6px 10px;border:1px solid #ccc'>Predicted</th>{header}</tr>"
        f"{rows}</table>"
    )


def build_summary_table_html(lead_results: dict[str, dict]) -> str:
    """
    Simplified table: average probability of each class across all leads
    combined (a quick overall glance, no per-lead detail) -- for fast
    field-study review.
    """
    all_classes = sorted({c for r in lead_results.values() for c in r["probs"]})
    n_leads = len(lead_results)
    avg_probs = {
        c: sum(r["probs"].get(c, 0.0) for r in lead_results.values()) / n_leads
        for c in all_classes
    }
    overall_predicted = max(avg_probs, key=avg_probs.get)
    rows = "".join(
        f"<tr><td style='padding:6px 10px;border:1px solid #ccc'>{c}</td>"
        f"<td style='padding:6px 10px;border:1px solid #ccc;text-align:center'>{avg_probs[c] * 100:.1f}%</td></tr>"
        for c in sorted(avg_probs, key=avg_probs.get, reverse=True)
    )
    return (
        f"<p style='font-family:Arial,sans-serif;font-size:14px'>"
        f"<b>Overall suggested result: {overall_predicted}</b></p>"
        "<table style='border-collapse:collapse;font-family:Arial,sans-serif;font-size:13px'>"
        "<tr><th style='padding:6px 10px;border:1px solid #ccc'>Class</th>"
        "<th style='padding:6px 10px;border:1px solid #ccc'>Percentage (average across leads)</th></tr>"
        f"{rows}</table>"
    )


# ============================================================
# 4) تجميع وإرسال الإيميل
# ============================================================
def generate_and_send_report(smtp_config: SMTPConfig, recipient_email: str,
                              patient_label: str, lead_results: dict[str, dict],
                              pre: int = 100) -> None:
    """
    lead_results: {اسم القطب: {"predicted": الفئة, "probs": {...},
                                "beat": np.ndarray, "ref_min": np.ndarray,
                                "ref_max": np.ndarray, "cutoff": int}}
    (نفس المخرجات المتوفرة أصلاً بعد استدعاء LeadModel.predict_beat لكل قطب --
    فقط أضِف "beat"/"ref_min"/"ref_max"/"cutoff" من نفس الكائن LeadModel
    المستخدَم بالتصنيف، هذي القيم أصلاً محفوظة عنده كـ self.ref_min/ref_max).

    يبني شكلاً لكل قطب + الجدولين، ويُرسل إيميل HTML واحد بكل شيء مضمَّناً.
    """
    images_cid = {}
    images_html = ""
    for i, (lead, r) in enumerate(lead_results.items()):
        cid = f"lead_fig_{i}"
        png_bytes = render_lead_figure(lead, r["beat"], r["ref_min"], r["ref_max"],
                                        r["cutoff"], pre, r["predicted"])
        images_cid[cid] = png_bytes
        images_html += f"<img src='cid:{cid}' style='max-width:600px;display:block;margin:8px 0'/>"

    detailed_table = build_detailed_table_html(lead_results)
    summary_table = build_summary_table_html(lead_results)

    html_body = f"""
    <html><head><meta charset="utf-8"></head><body style="font-family:Arial,sans-serif">
      <h2>ECG Recording Classification Report — {patient_label}</h2>

      <h3>Summary result</h3>
      {summary_table}

      <h3>Detailed table (per lead)</h3>
      {detailed_table}

      <h3>Charts (each lead: patient beat vs. normal range)</h3>
      {images_html}

      <p style="color:#888;font-size:12px">Automated report -- for review and field validation only; does not replace direct clinical assessment.</p>
    </body></html>
    """

    msg = MIMEMultipart("related")
    msg["Subject"] = f"ECG Classification Report — {patient_label}"
    msg["From"] = smtp_config.sender_email
    msg["To"] = recipient_email
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    for cid, png_bytes in images_cid.items():
        img = MIMEImage(png_bytes, _subtype="png")
        img.add_header("Content-ID", f"<{cid}>")
        img.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
        msg.attach(img)

    with smtplib.SMTP(smtp_config.host, smtp_config.port) as server:
        if smtp_config.use_tls:
            server.starttls()
        server.login(smtp_config.username, smtp_config.password)
        server.send_message(msg)
