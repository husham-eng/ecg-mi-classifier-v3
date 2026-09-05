"""
app.py — تطبيق الويب الرئيسي
=================================
واجهة ويب بسيطة (Flask) لتصنيف نوع احتشاء عضلة القلب من:
  - ملف إشارة رقمية (CSV/TXT، عمود واحد من القيم)
  - أو صورة مخطط ECG (JPG/PNG) مصوَّرة أو ممسوحة ضوئياً

يدعم إدخال أكثر من قطب من الأقطاب الأربعة المدعومة (Lead I, aVR, V2, V6)
في نفس الطلب؛ يُدمَج القرار النهائي تلقائياً عبر تصويت مرجّح إن توفر
أكثر من قطب.

⚠️ هذا نموذج أولي بحثي وليس جهازاً طبياً معتمداً. النتائج للمساعدة على
تحديد الأولوية فقط ويجب دائماً تأكيدها طبياً.
"""

import os
import io
import json
import uuid
import base64
import tempfile
from functools import wraps
from datetime import datetime, timezone

import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, session, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

from ecg_pipeline import (classify_patient, classify_from_image, classify_lead_signal,
                           combine_lead_probabilities, SUPPORTED_LEADS)
from ecg_pipeline.panel_detector import detect_panel_leads
from ecg_pipeline.email_report import generate_and_send_report, SMTPConfig
from translations import get_translation, DEFAULT_LANG

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB حد أقصى لكل طلب

# ⚠️ لأمان حقيقي: اضبط هذين المتغيّرين كمتغيّرات بيئة على منصة النشر
# (Render: تبويب Environment)، لا تعتمد على القيم الافتراضية بالإنتاج.
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me-on-render")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "changeme123")

# ⚠️ ميزة تقرير الإيميل (جلسة الدراسة الميدانية): تحتاج متغيّرات بيئة
# ECG_SMTP_HOST / ECG_SMTP_PORT / ECG_SMTP_USERNAME / ECG_SMTP_PASSWORD
# (راجع توثيق SMTPConfig.from_env بـ ecg_pipeline/email_report.py). إن لم
# تُضبَط، تبقى الميزة معطَّلة تلقائياً (لا خطأ يوقف التطبيق) -- أي طلب فيه
# doctor_email سيُرجع تحذيراً بالنتيجة بدل إرسال فعلي، حتى تُضبَط الإعدادات.
try:
    _SMTP_CONFIG = SMTPConfig.from_env()
except KeyError:
    _SMTP_CONFIG = None

# ⚠️ تنبيه مهم: هذا المجلد على قرص مؤقت (Ephemeral) بمعظم منصات الاستضافة
# المجانية (بما فيها Render Free) — يُمسَح بالكامل عند كل إعادة نشر أو
# "نوم" الخادم لفترة طويلة. مناسب لمراجعة قصيرة المدى فقط، وليس أرشيفاً
# دائماً، إلى أن يُضاف تخزين خارجي دائم (قاعدة بيانات/تخزين سحابي).
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
SIGNAL_EXTENSIONS = {".csv", ".txt"}


def load_signal_file(file_storage) -> np.ndarray:
    """يقرأ ملف إشارة (عمود واحد من الأرقام، مع أو بدون رأس نصي)."""
    content = file_storage.read().decode("utf-8", errors="ignore")
    values = []
    for line in content.splitlines():
        line = line.strip().split(",")[0].split("\t")[0]
        try:
            values.append(float(line))
        except ValueError:
            continue
    return np.array(values)


def log_attempt(kind: str, raw_files: dict[str, tuple[bytes, str]], result: dict) -> str:
    """
    يحفظ محاولة (تصنيف أو اكتشاف لوحة) بمجلد logs/ لمراجعتها لاحقاً عبر
    لوحة التحكم: الملفات المرفوعة كما هي + نتيجة المعالجة كـJSON + وقت
    الطلب. راجع التنبيه أعلى الملف بخصوص ديمومة هذا التخزين.
    """
    timestamp = datetime.now(timezone.utc)
    attempt_id = timestamp.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    attempt_dir = os.path.join(LOGS_DIR, attempt_id)
    os.makedirs(attempt_dir, exist_ok=True)

    for field_name, (content, original_filename) in raw_files.items():
        safe_name = secure_filename(f"{field_name}_{original_filename}") or f"{field_name}.bin"
        with open(os.path.join(attempt_dir, safe_name), "wb") as f:
            f.write(content)

    meta = {"kind": kind, "timestamp": timestamp.isoformat(), "result": result}
    with open(os.path.join(attempt_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    return attempt_id


@app.route("/", methods=["GET"])
def index():
    lang = request.args.get("lang", DEFAULT_LANG)
    t = get_translation(lang)
    return render_template("index.html", leads=SUPPORTED_LEADS, t=t)


@app.route("/detect_panel", methods=["POST"])
def detect_panel():
    """
    يستقبل صورة لوحة ECG كاملة (12 قطباً)، يكتشف ويقصّ الأقطاب الأربعة
    المدعومة تلقائياً (LeadI, aVR, V2, V6)، ويُرجعها كصور مصغّرة (base64)
    مع مستوى ثقة لكل واحدة — لعرضها بخطوة تأكيد بصرية قبل التصنيف
    النهائي (لا يُصنَّف شيء هنا، فقط اكتشاف وتقطيع).
    """
    file = request.files.get("panel_image")
    if not file or file.filename == "":
        return jsonify({"error": "لم يتم رفع أي صورة."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in IMAGE_EXTENSIONS:
        return jsonify({"error": f"صيغة ملف غير مدعومة لصورة اللوحة الكاملة: {ext}"}), 400

    raw_bytes = file.read()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        detected = detect_panel_leads(tmp_path)
    except Exception as e:
        error_result = {"error": f"تعذّر اكتشاف اللوحة — تحقق من أنها صورة تخطيط قياسية واضحة. ({e})"}
        log_attempt("detect_panel", {"panel_image": (raw_bytes, file.filename)}, error_result)
        return jsonify(error_result), 400
    finally:
        os.unlink(tmp_path)

    # نحتاج فقط الأقطاب الأربعة المدعومة فعلياً بالتطبيق
    panel_to_project = {"I": "LeadI", "aVR": "aVR", "V2": "V2", "V6": "V6"}
    leads_out = {}
    for panel_name, project_name in panel_to_project.items():
        info = detected.get(panel_name)
        if info is None:
            continue
        crop = info["crop"]
        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            continue
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        leads_out[project_name] = {
            "image_b64": b64,
            "confidence": info["confidence"],  # confirmed / weak / not_found
        }

    if not leads_out:
        error_result = {"error": "تعذّر اكتشاف أي قطب مدعوم بهذي الصورة."}
        log_attempt("detect_panel", {"panel_image": (raw_bytes, file.filename)}, error_result)
        return jsonify(error_result), 400

    # نسجّل الصورة الأصلية + ملخّص الثقة لكل قطب (لا الصور المقصوصة base64،
    # لتوفير المساحة — يمكن استنتاجها لاحقاً من الصورة الأصلية عند الحاجة).
    log_summary = {"leads_confidence": {k: v["confidence"] for k, v in leads_out.items()}}
    log_attempt("detect_panel", {"panel_image": (raw_bytes, file.filename)}, log_summary)

    return jsonify({"leads": leads_out})


@app.route("/classify", methods=["POST"])
def classify():
    per_lead_results = {}
    errors = {}
    raw_files_for_log = {}  # {field_name: (raw_bytes, original_filename)}

    # قطب Lead II اختياري: لو رُفع، يُستخدم فقط كمرجع محازاة موحّد لمواقع R
    # عبر الأقطاب المُدخَلة كإشارات رقمية (لا يُصنَّف بنفسه، ولا يُطبَّق
    # حالياً على مسار الصور — كل صورة لها توقيتها الخاصة من تحليلها).
    reference_lead_signal = None
    ref_file = request.files.get("file_II")
    if ref_file and ref_file.filename != "":
        ref_raw_bytes = ref_file.read()
        raw_files_for_log["II"] = (ref_raw_bytes, ref_file.filename)
        ref_ext = os.path.splitext(ref_file.filename)[1].lower()
        if ref_ext in SIGNAL_EXTENSIONS:
            ref_fs = float(request.form.get("fs_II", 500))
            ref_raw = np.array([float(x) for x in ref_raw_bytes.decode("utf-8", errors="ignore").splitlines()
                                 if x.strip().replace(".", "", 1).replace("-", "", 1).isdigit()])
            if len(ref_raw) >= ref_fs:
                reference_lead_signal = (ref_raw, ref_fs)
            else:
                errors["II"] = "إشارة Lead II المرجعية قصيرة جداً — تم تجاهلها."
        else:
            errors["II"] = "Lead II المرجعي مدعوم كملف إشارة رقمية (CSV/TXT) فقط حالياً."

    external_r_locs = None
    if reference_lead_signal is not None:
        from ecg_pipeline.preprocessing import denoise, remove_baseline_highpass, detect_r_peaks
        ref_raw, ref_fs = reference_lead_signal
        ref_clean = remove_baseline_highpass(denoise(ref_raw, ref_fs), ref_fs)
        external_r_locs = detect_r_peaks(ref_clean, ref_fs, polarity_robust=True)

    for lead in SUPPORTED_LEADS:
        file = request.files.get(f"file_{lead}")
        if not file or file.filename == "":
            continue

        raw_bytes = file.read()
        raw_files_for_log[lead] = (raw_bytes, file.filename)
        ext = os.path.splitext(file.filename)[1].lower()

        if ext in IMAGE_EXTENSIONS:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(raw_bytes)
                tmp_path = tmp.name
            try:
                result = classify_from_image(tmp_path, lead)
            finally:
                os.unlink(tmp_path)
            if "error" in result:
                errors[lead] = result["error"]
            else:
                per_lead_results[lead] = result

        elif ext in SIGNAL_EXTENSIONS:
            fs = float(request.form.get(f"fs_{lead}", 500))
            values = []
            for line in raw_bytes.decode("utf-8", errors="ignore").splitlines():
                cell = line.strip().split(",")[0].split("\t")[0]
                try:
                    values.append(float(cell))
                except ValueError:
                    continue
            raw = np.array(values)
            if len(raw) < fs:  # أقل من ثانية واحدة من البيانات
                errors[lead] = "الإشارة قصيرة جداً (أقل من ثانية واحدة)."
                continue
            result = classify_lead_signal(raw, fs, lead, external_r_locs=external_r_locs)
            if "error" in result:
                errors[lead] = result["error"]
            else:
                per_lead_results[lead] = result
        else:
            errors[lead] = f"صيغة ملف غير مدعومة: {ext}"

    if not per_lead_results:
        error_result = {"error": "لم يتم رفع أي قطب صالح.", "details": errors}
        if raw_files_for_log:
            log_attempt("classify", raw_files_for_log, error_result)
        return jsonify(error_result), 400

    # دمج فعلي لكل الأقطاب المتوفرة (صور و/أو إشارات معاً) بتصويت مرجّح واحد،
    # بغض النظر عن نوع المصدر لكل قطب — هذا يصلح الفجوة التي كانت تمنع
    # دمج الصور المتعددة سابقاً (كانت تُرجع خطأً بدل قرار نهائي).
    result = combine_lead_probabilities(per_lead_results)
    if "error" not in result:
        result["reference_lead_alignment_used"] = reference_lead_signal is not None
    result["errors"] = errors

    # ⚠️ ميزة تقرير الإيميل (جلسة الدراسة الميدانية): نستخرج بيانات الرسم
    # البياني (النبضة الفعلية + المدى الطبيعي + نقطة القطع) من per_lead_results
    # *قبل* حذفها لاحقاً من النتيجة (numpy arrays غير قابلة للتحويل لـJSON
    # مباشرة عبر jsonify -- راجع الحذف أدناه).
    doctor_email = (request.form.get("doctor_email") or "").strip()
    if doctor_email and "error" not in result:
        lead_results_for_email = {}
        for lead, r in per_lead_results.items():
            if "error" in r or "representative_beat" not in r:
                continue
            lead_results_for_email[lead] = {
                "predicted": max(r["probabilities"], key=r["probabilities"].get),
                "probs": r["probabilities"],
                "beat": r["representative_beat"],
                "ref_min": r["ref_min"],
                "ref_max": r["ref_max"],
                "cutoff": r["representative_cutoff"],
            }
        if lead_results_for_email:
            if _SMTP_CONFIG is None:
                result["email_status"] = ("لم يُرسَل: إعدادات SMTP غير مضبوطة على السيرفر "
                                           "(راجع متغيّرات ECG_SMTP_* بالبيئة).")
            else:
                try:
                    generate_and_send_report(_SMTP_CONFIG, doctor_email,
                                              patient_label=f"recording_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}",
                                              lead_results=lead_results_for_email)
                    result["email_status"] = f"أُرسل تقرير مفصَّل إلى {doctor_email}."
                except Exception as e:
                    result["email_status"] = f"تعذّر إرسال الإيميل: {e}"

    # نحذف الآن أي مصفوفات numpy خام من النتيجة قبل log_attempt/jsonify --
    # كانت ضرورية فقط للحظة إعداد الإيميل أعلاه، وغير قابلة للتحويل لـJSON.
    for r in per_lead_results.values():
        r.pop("representative_beat", None)
        r.pop("ref_min", None)
        r.pop("ref_max", None)

    log_attempt("classify", raw_files_for_log, result)
    return jsonify(result)


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("is_admin"):
            return redirect(url_for("admin_login"))
        return view_func(*args, **kwargs)
    return wrapper


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    error = None
    if request.method == "POST":
        if request.form.get("password") == ADMIN_PASSWORD:
            session["is_admin"] = True
            return redirect(url_for("admin_dashboard"))
        error = "كلمة المرور غير صحيحة."
    return render_template("admin_login.html", error=error)


@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("admin_login"))


@app.route("/admin")
@login_required
def admin_dashboard():
    attempts = []
    if os.path.isdir(LOGS_DIR):
        for attempt_id in sorted(os.listdir(LOGS_DIR), reverse=True):
            attempt_dir = os.path.join(LOGS_DIR, attempt_id)
            meta_path = os.path.join(attempt_dir, "meta.json")
            if not os.path.isfile(meta_path):
                continue
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            files = sorted(fn for fn in os.listdir(attempt_dir) if fn != "meta.json")
            image_files = [fn for fn in files if os.path.splitext(fn)[1].lower() in IMAGE_EXTENSIONS]
            attempts.append({
                "id": attempt_id,
                "kind": meta.get("kind"),
                "timestamp": meta.get("timestamp"),
                "result": meta.get("result", {}),
                "files": files,
                "image_files": image_files,
            })
    return render_template("admin_dashboard.html", attempts=attempts, logs_dir=LOGS_DIR)


@app.route("/admin/file/<attempt_id>/<filename>")
@login_required
def admin_file(attempt_id, filename):
    # حماية من Path Traversal: نتحقق أن المعرّف والاسم موجودان حرفياً بقائمة
    # مجلد logs/ الفعلية قبل تقديم أي ملف.
    safe_attempt_id = secure_filename(attempt_id)
    attempt_dir = os.path.join(LOGS_DIR, safe_attempt_id)
    if not os.path.isdir(attempt_dir):
        return "غير موجود", 404
    safe_filename = secure_filename(filename)
    if safe_filename not in os.listdir(attempt_dir):
        return "غير موجود", 404
    return send_from_directory(attempt_dir, safe_filename)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
