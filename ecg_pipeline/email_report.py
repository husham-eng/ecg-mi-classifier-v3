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

⚠️ إصلاح جوهري (جلسة تشخيص فشل الإرسال بالإنتاج): النسخة الأولى استخدمت
SMTP عادي (smtplib) عبر المنفذ 587 -- اكتُشِف بالتجربة الفعلية أن Render
(ومعظم منصات الاستضافة السحابية المجانية) **يحجب المنافذ الصادرة الخاصة
بالبريد** (25/465/587) لمنع إساءة الاستخدام كمصدر سبام، بغض النظر عن صحة
بيانات الاعتماد. النتيجة: محاولة الاتصال تتعلّق (Hang) بلا خطأ واضح، لين
تتجاوز مهلة gunicorn (WORKER TIMEOUT) ويُقتَل الطلب بالكامل.

**الحل**: إرسال عبر واجهة SendGrid البرمجية (HTTPS، منفذ 443 -- غير محجوب
أبداً بأي منصة سحابية) بدل SMTP كلياً. الصور أيضاً صارت مُضمَّنة كـ Base64
Data URI مباشرة بالـHTML (بدل CID/MIME منفصل) -- أبسط وتعمل مع أي مزوّد
بريد إلكتروني بلا اعتماد على دعمه الخاص للصور المضمَّنة.

يحتاج:
  - حساب SendGrid مجاني (100 إيميل/يوم مجاناً بشكل دائم).
  - تفعيل "Single Sender Verification" لعنوان بريد واحد فقط (بريدك
    الشخصي مثلاً) من لوحة SendGrid -- يستغرق دقيقتين، لا يحتاج نطاقاً
    (Domain) كاملاً.
  - مفتاح API من SendGrid (Settings → API Keys → Create API Key).
  لا تُخزَّن بيانات الاعتماد داخل الكود إطلاقاً -- مرَّرها متغيرات بيئة
  (راجع EmailAPIConfig.from_env أدناه).
"""

from __future__ import annotations
import base64
import io
import os
from dataclasses import dataclass

import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")  # لا حاجة لواجهة رسومية على سيرفر السحابة
import matplotlib.pyplot as plt

SENDGRID_ENDPOINT = "https://api.sendgrid.com/v3/mail/send"


# ============================================================
# 1) إعدادات SendGrid (من متغيرات البيئة -- لا مفاتيح بالكود)
# ============================================================
@dataclass
class EmailAPIConfig:
    api_key: str
    sender_email: str

    @classmethod
    def from_env(cls) -> "EmailAPIConfig":
        """
        يقرأ الإعدادات من متغيرات البيئة:
          ECG_SENDGRID_API_KEY  (من لوحة SendGrid: Settings → API Keys)
          ECG_SENDGRID_SENDER   (بريد "Single Sender" الذي فعّلته بحسابك)
        """
        return cls(
            api_key=os.environ["ECG_SENDGRID_API_KEY"],
            sender_email=os.environ["ECG_SENDGRID_SENDER"],
        )


# ============================================================
# 2) شكل بياني لكل قطب (نبضة المريض الفعلية فوق المدى الطبيعي)
# ============================================================
def render_lead_figure_base64(lead: str, beat: np.ndarray, ref_min: np.ndarray, ref_max: np.ndarray,
                               cutoff: int, pre: int, predicted_class: str) -> str:
    """
    يرسم نبضة مريض واحد فوق المدى الطبيعي المظلَّل لنفس القطب، مع تظليل
    رمادي لأي جزء مقنَّع (تجاوز نقطة القطع الديناميكية). يرجع الصورة
    كسلسلة Base64 (بلا بادئة data:) جاهزة للتضمين المباشر بالـHTML.

    ⚠️ نصوص الشكل بالإنجليزية عمداً: matplotlib يعرض النص العربي معكوساً
    ومفكّكاً بلا مكتبات تشكيل إضافية (arabic_reshaper + python-bidi) غير
    مضمونة التوفّر بكل بيئة سحابية.
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
    return base64.b64encode(buf.read()).decode("ascii")


# ============================================================
# 3) الجدولان (تفصيلي + مبسَّط) -- بلا تغيير عن النسخة السابقة
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
# 4) تجميع وإرسال الإيميل (عبر SendGrid API، لا SMTP)
# ============================================================
def generate_and_send_report(email_config: EmailAPIConfig, recipient_email: str,
                              patient_label: str, lead_results: dict[str, dict],
                              pre: int = 100, timeout_seconds: int = 15) -> None:
    """
    lead_results: {اسم القطب: {"predicted": الفئة, "probs": {...},
                                "beat": np.ndarray, "ref_min": np.ndarray,
                                "ref_max": np.ndarray, "cutoff": int}}

    يبني شكلاً لكل قطب (مُضمَّن Base64 مباشرة بالـHTML) + الجدولين،
    ويُرسل عبر SendGrid API (طلب HTTPS واحد، منفذ 443). يرفع استثناءً
    عادياً عند أي فشل (شبكة، رفض المفتاح، تجاوز الحصة اليومية...) --
    المستدعي (app.py) مسؤول عن الإمساك به وتسجيل الحالة دون إسقاط الطلب
    الأساسي (التصنيف نفسه يجب أن ينجح حتى لو فشل الإيميل).

    timeout_seconds: مهلة قصوى لطلب SendGrid نفسه (منفصلة تماماً عن أي
    قيد SMTP سابق) -- تحمي من تعليق الطلب الأساسي حتى لو تعطّلت خدمة
    SendGrid نفسها لأي سبب.
    """
    images_html = ""
    for lead, r in lead_results.items():
        b64_png = render_lead_figure_base64(lead, r["beat"], r["ref_min"], r["ref_max"],
                                             r["cutoff"], pre, r["predicted"])
        images_html += (
            f"<img src='data:image/png;base64,{b64_png}' "
            f"style='max-width:600px;display:block;margin:8px 0'/>"
        )

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

    payload = {
        "personalizations": [{"to": [{"email": recipient_email}]}],
        "from": {"email": email_config.sender_email},
        "subject": f"ECG Classification Report — {patient_label}",
        "content": [{"type": "text/html", "value": html_body}],
    }
    headers = {
        "Authorization": f"Bearer {email_config.api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(SENDGRID_ENDPOINT, json=payload, headers=headers, timeout=timeout_seconds)
    if response.status_code >= 400:
        raise RuntimeError(f"SendGrid rejected the request ({response.status_code}): {response.text[:500]}")
