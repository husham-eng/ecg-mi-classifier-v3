"""
ecg_pipeline.pipeline
========================
نقطة الدخول الموحّدة للتطبيق: من إشارة خام (أو صورة) لكل قطب متوفر،
إلى قرار تصنيف نهائي مدمج عبر تصويت مرجّح (Weighted Voting) بين كل
الأقطاب المتاحة.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

from .preprocessing import process_raw_signal, denoise, remove_baseline_highpass, detect_r_peaks
from .classifier import LeadModel, MODELS_DIR
from .image_digitizer import extract_trace_from_image, calibrate_signal

SUPPORTED_LEADS = ["LeadI", "aVR", "V2", "V6"]
REFERENCE_LEAD = "II"  # قطب اختياري إضافي: لو رُفع، يُستخدم فقط لتوحيد محازاة النبضات

# أوزان تصويت مرجّح ثابتة، مبنية على Macro-F1 الموثّق لكل قطب (README.md،
# تقييم مستقل على مستوى المريض). V2 وV6 أثبتا أداءً أفضل باستمرار عبر
# كل تجارب المشروع، فمن المنطقي إعطاؤهما تأثيراً أكبر بالقرار المدمج
# بدل معاملتهما بنفس وزن Lead I الأضعف أداءً.
DEFAULT_LEAD_WEIGHTS = {
    "V2": 0.423,
    "V6": 0.395,
    "aVR": 0.338,
    "LeadI": 0.312,
}

_loaded_models: dict[str, LeadModel] = {}


def get_model(lead: str) -> LeadModel:
    if lead not in _loaded_models:
        _loaded_models[lead] = LeadModel.load(MODELS_DIR / lead)
    return _loaded_models[lead]


def classify_lead_signal(raw_signal: np.ndarray, fs: float, lead: str,
                          beat_index: int | None = None,
                          external_r_locs: np.ndarray | None = None) -> dict:
    """
    يصنّف قطباً واحداً من إشارة خام مستمرة.

    external_r_locs: مواقع R جاهزة (عادة من قطب مرجعي مثل Lead II إن
    توفّر) بدل اكتشافها من هذا القطب بمفرده. فحص فعلي على بيانات PTB-XL
    أظهر أن الاعتماد على قطب مرجعي موحّد لتحديد مواقع R يحسّن Macro-F1
    بشكل ملموس، خصوصاً بقطب aVR حيث ينعكس مركّب QRS غالباً (القمة
    السالبة أوضح من الموجبة)، ما يجعل اكتشافه المستقل أقل موثوقية من
    الاعتماد على قطب أوضح كمرجع. إن لم يُمرَّر، يُكتشَف تلقائياً من هذا
    القطب نفسه (بمقاومة انعكاس القطبية المفعّلة افتراضياً في detect_r_peaks).

    beat_index=None (الافتراضي): يصنّف كل نبضة مكتشفة على حدة ثم يُرجع
    متوسط الاحتمالات عبرها (انظر توثيق n_beats_used وbeat_agreement).
    beat_index=<رقم>: نبضة واحدة محددة صراحة (اختبارات/تصحيح أخطاء).

    ⚠️ إصلاح جوهري (جلسة تشخيص طول النافذة): process_raw_signal يرجع
    الآن نقطة قطع ديناميكية لكل نبضة (cutoffs) بجانب النبضات ومواقع R،
    تُمرَّر إجبارياً لـ predict_beat حتى لا يقارن أي جزء من نبضة يتجاوز
    بداية النبضة التالية فعلياً (خصوصاً عند مرضى نبضهم سريع) بالمرجع.

    النتيجة تتضمّن أيضاً "representative_beat" (النبضة المستخدَمة فعلياً
    فأول نبضة مكتشفة، أو المحدَّدة عبر beat_index) + المدى الطبيعي لهذا
    القطب (ref_min/ref_max) ونقطة قطعها -- جاهزة مباشرة لأي رسم بياني
    لاحق (مثلاً تقرير الإيميل عبر ecg_pipeline.email_report) بلا حاجة
    لإعادة استخراجها من الصفر.
    """
    model = get_model(lead)
    pre = model.window_len * 200 // 800  # نسبة قياسية مطابقة لنافذة التدريب (±400ms عند fs=1000)
    post = model.window_len - pre
    beats, r_locs, cutoffs = process_raw_signal(raw_signal, fs, pre=pre, post=post,
                                                 external_r_locs=external_r_locs)
    if len(beats) == 0:
        return {"error": "لم يتم اكتشاف أي نبضة صالحة بهذا القطب — تحقق من جودة الإشارة."}

    if beat_index is not None:
        idx = min(beat_index, len(beats) - 1)
        probs = model.predict_beat(beats[idx], cutoff=cutoffs[idx])
        return {"lead": lead, "n_beats_detected": len(beats), "n_beats_used": 1,
                "probabilities": probs,
                "representative_beat": beats[idx], "representative_cutoff": int(cutoffs[idx]),
                "ref_min": model.ref_min, "ref_max": model.ref_max}

    all_probs = [model.predict_beat(b, cutoff=c) for b, c in zip(beats, cutoffs)]
    classes = set()
    for p in all_probs:
        classes.update(p.keys())
    mean_probs = {c: float(np.mean([p.get(c, 0.0) for p in all_probs])) for c in classes}

    per_beat_top = [max(p, key=p.get) for p in all_probs]
    most_common = max(set(per_beat_top), key=per_beat_top.count)
    agreement = per_beat_top.count(most_common) / len(per_beat_top)

    return {
        "lead": lead,
        "n_beats_detected": len(beats),
        "n_beats_used": len(beats),
        "beat_agreement": round(agreement, 3),
        "probabilities": mean_probs,
        # نبضة تمثيلية (الأولى المكتشفة) + مدى القطب الطبيعي، لأي عرض بياني لاحق
        "representative_beat": beats[0], "representative_cutoff": int(cutoffs[0]),
        "ref_min": model.ref_min, "ref_max": model.ref_max,
    }


def combine_lead_probabilities(per_lead_results: dict[str, dict],
                                lead_weights: dict[str, float] | None = None) -> dict:
    """
    يدمج نتائج أي مجموعة من الأقطاب (سواء أتت من classify_lead_signal أو
    classify_from_image أو مزيج من الاثنين) بتصويت مرجّح فعلي، ويُرجع
    القرار النهائي. هذي الدالة العامة المشتركة التي يعتمد عليها كل من
    classify_patient (للإشارات) وapp.py (لدمج الصور المتعددة، الذي كان
    غير مُطبَّق سابقاً — راجع ملاحظة إصلاح الجلسة أدناه).

    per_lead_results: قاموس {اسم القطب: نتيجة classify_lead_signal/
    classify_from_image} — أي نتيجة فيها مفتاح "probabilities" صالحة،
    وأي نتيجة فيها "error" تُستبعد من الدمج لكنها تبقى بالنتيجة النهائية
    ضمن per_lead للشفافية.
    """
    weights = dict(DEFAULT_LEAD_WEIGHTS)
    if lead_weights:
        weights.update(lead_weights)

    all_classes = set()
    for result in per_lead_results.values():
        if "error" not in result:
            all_classes.update(result["probabilities"].keys())

    if not per_lead_results:
        return {"error": "لا يوجد قطب مدعوم ضمن المدخلات (المدعوم حالياً: Lead I, aVR, V2, V6)."}

    combined = {c: 0.0 for c in all_classes}
    total_weight = 0.0
    n_valid = 0
    weights_used = {}
    for lead, result in per_lead_results.items():
        if "error" in result:
            continue
        w = weights.get(lead, 1.0)
        weights_used[lead] = w
        total_weight += w
        n_valid += 1
        for c, p in result["probabilities"].items():
            combined[c] += w * p

    if n_valid == 0:
        return {"error": "تعذّر استخراج نبضات صالحة من أي قطب مُدخَل.",
                "per_lead": per_lead_results}

    combined = {c: v / total_weight for c, v in combined.items()}
    final_class = max(combined, key=combined.get)

    return {
        "final_classification": final_class,
        "confidence": round(combined[final_class], 3),
        "combined_probabilities": {c: round(v, 3) for c, v in combined.items()},
        "per_lead": per_lead_results,
        "n_leads_used": n_valid,
        "lead_weights_used": {k: round(v, 3) for k, v in weights_used.items()},
        "disclaimer": (
            "هذا تصنيف أوّلي آلي بغرض المساعدة على تحديد الأولوية، وليس تشخيصاً طبياً نهائياً. "
            "يجب دائماً تأكيد النتيجة عبر تقييم طبي مختص."
        ),
    }


def classify_patient(signals_by_lead: dict[str, tuple[np.ndarray, float]],
                      lead_weights: dict[str, float] | None = None,
                      reference_lead_signal: tuple[np.ndarray, float] | None = None) -> dict:
    """
    نقطة الدخول الرئيسية لإشارات رقمية (CSV/TXT): تأخذ قاموس {اسم القطب:
    (إشارة خام، معدل العينات)} لأي مجموعة فرعية من الأقطاب المدعومة
    (LeadI, aVR, V2, V6)، وتُرجع القرار النهائي المدمج بتصويت مرجّح فعلي
    بين الأقطاب (عبر combine_lead_probabilities).

    reference_lead_signal: اختياري — (إشارة Lead II الخام، معدل عيّناتها)
    إن توفّرت لدى المستخدم (مثلاً رفع تخطيط كامل 12 قطباً رغم أن النموذج
    يستخدم 4 منها فقط). تُستخدم فقط لاكتشاف مواقع R موحّدة تُطبَّق على
    كل الأقطاب المُدخَلة، بدل اكتشاف كل قطب مواقعه بمفرده — يحسّن الدقة
    بشكل ملموس خصوصاً لقطب aVR (راجع توثيق classify_lead_signal). يُشترط
    أن تكون إشارة Lead II بنفس معدل عينات وطول أقطاب المريض المُدخَلة
    (نفس جلسة التسجيل).

    lead_weights: قاموس اختياري لتخصيص أوزان الأقطاب (افتراضياً
    DEFAULT_LEAD_WEIGHTS المبنية على Macro-F1 الموثّق لكل قطب).
    """
    external_r_locs = None
    if reference_lead_signal is not None:
        ref_raw, ref_fs = reference_lead_signal
        ref_clean = remove_baseline_highpass(denoise(ref_raw, ref_fs), ref_fs)
        external_r_locs = detect_r_peaks(ref_clean, ref_fs, polarity_robust=True)

    per_lead_results = {}
    for lead, (raw, fs) in signals_by_lead.items():
        if lead not in SUPPORTED_LEADS:
            continue
        per_lead_results[lead] = classify_lead_signal(raw, fs, lead, external_r_locs=external_r_locs)

    result = combine_lead_probabilities(per_lead_results, lead_weights)
    if "error" not in result:
        result["reference_lead_alignment_used"] = reference_lead_signal is not None
    return result


def classify_from_image(image_path: str, lead: str) -> dict:
    """يحوّل صورة مخطط (لقطب واحد) إلى إشارة، ثم يصنّفها عبر نفس الـpipeline."""
    pixel_signal, valid_mask = extract_trace_from_image(image_path)
    if valid_mask.mean() < 0.5:
        return {"error": "جودة استخراج الأثر من الصورة منخفضة جداً (أقل من 50% من الأعمدة واضحة). "
                          "جرّب صورة أوضح أو أقل ميلاناً/انعكاساً."}
    mv_signal, meta = calibrate_signal(pixel_signal, image_path)
    result = classify_lead_signal(mv_signal, meta["target_fs"], lead)
    result["calibration_meta"] = meta
    result["extraction_quality"] = round(float(valid_mask.mean()), 3)
    return result
