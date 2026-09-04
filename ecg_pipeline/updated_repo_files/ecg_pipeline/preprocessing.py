
"""
ecg_pipeline.preprocessing
============================
خطوات معالجة الإشارة الأساسية: إزالة الضوضاء، إزالة الانحراف الأساسي،
كشف موجة R، وتقطيع/محاذاة النبضات.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import ellip, filtfilt, find_peaks


def denoise(x: np.ndarray, fs: float, cutoff_hz: float = 75.0) -> np.ndarray:
    """فلتر تمرير منخفض إهليلجي (رتبة 7) لإزالة الضوضاء عالية التردد."""
    nyq = fs / 2.0
    b, a = ellip(N=7, rp=1, rs=60, Wn=cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, x)


def remove_baseline(x: np.ndarray, degree: int = 6) -> np.ndarray:
    """إزالة الانحراف الأساسي (Baseline Wander) عبر polyfit."""
    n = len(x)
    t = np.arange(1, n + 1, dtype=float)
    return x - np.polyval(np.polyfit(t, x, degree), t)


def detect_r_peaks(clean: np.ndarray, fs: float, min_distance_s: float = 0.4,
                    polarity_robust: bool = True) -> np.ndarray:
    """كشف قمم R بعتبة تكيّفية (50% من المئين 99)."""
    search_signal = np.abs(clean) if polarity_robust else clean
    thresh = 0.5 * np.percentile(search_signal, 99)
    peaks, _ = find_peaks(search_signal, height=thresh, distance=int(min_distance_s * fs))
    return peaks


def detect_r_consensus(candidates_per_lead: dict[str, np.ndarray], n_leads_total: int,
                        cluster_window_samples: int, min_agreement_fraction: float = 0.5) -> np.ndarray:
    """
    ⚠️ إصلاح جوهري (موثَّق بمقالة المشروع): بدل الاعتماد على قطب واحد
    (Lead II) كمرجع موحّد لمحازاة كل الأقطاب — وهو نفسه قطب سفلي، غير
    موثوق تحديداً بفئات الاحتشاء السفلي (IL/IPL) — نستخدم إجماعاً عبر
    كل الأقطاب المتاحة بالتسجيل. كل قطب يكتشف مواقع R بشكل مستقل، ثم
    نجمّع الكشوفات المتقاربة زمنياً (تجميع تسلسلي بنافذة زمنية) ونأخذ
    الوسيط (median) لكل تجمّع يحظى بموافقة عدد كافٍ من الأقطاب. هذا
    يستبعد تأثير أي قطب واحد مشوَّه بسبب المرض نفسه، وأثبت تحسيناً
    كبيراً بجودة المحازاة (coherence_ratio) تجريبياً، خصوصاً بفئتي
    IL/IPL (راجع قسم "تصحيح المحازاة" بالمقالة للتفاصيل والأرقام).

    Parameters
    ----------
    candidates_per_lead : {اسم القطب: مصفوفة مواضع R المكتشفة بذلك القطب بشكل مستقل}
    n_leads_total : عدد الأقطاب الكلي المتاحة بالتسجيل (لحساب عتبة الإجماع)
    cluster_window_samples : أقصى مسافة (بالعينات) بين كشفين لاعتبارهما لنفس النبضة
    min_agreement_fraction : أقل نسبة أقطاب يجب أن توافق على موضع مرشّح ليُعتمَد
    """
    all_positions = []
    for positions in candidates_per_lead.values():
        all_positions.extend(positions.tolist())
    if not all_positions:
        return np.array([], dtype=int)

    all_positions = np.sort(np.array(all_positions))
    clusters, current = [], [all_positions[0]]
    for p in all_positions[1:]:
        if p - current[-1] <= cluster_window_samples:
            current.append(p)
        else:
            clusters.append(current)
            current = [p]
    clusters.append(current)

    min_votes = max(2, int(round(min_agreement_fraction * n_leads_total)))
    consensus = [int(np.median(c)) for c in clusters if len(c) >= min_votes]
    return np.array(sorted(consensus), dtype=int)


def extract_beats(x: np.ndarray, r_locs: np.ndarray, pre: int, post: int) -> np.ndarray:
    """يقتطع نافذة ثابتة (pre عينة قبل R، post عينة بعده) حول كل قمة R."""
    beats = []
    n = len(x)
    for r in r_locs:
        lo, hi = r - pre, r + post
        if lo >= 0 and hi <= n:
            beats.append(x[lo:hi])
    return np.array(beats)


def process_raw_signal(raw: np.ndarray, fs: float, pre: int = 200, post: int = 600,
                        cutoff_hz: float = 75.0,
                        external_r_locs: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    خط المعالجة الكامل من إشارة خام إلى نبضات مقطوعة ومحاذاة.

    external_r_locs: مواقع R جاهزة (يُفضَّل أن تكون ناتج detect_r_consensus
    بدل قطب مرجعي واحد فقط، عند توفّر تسجيل متعدد الأقطاب كاملاً وقت
    التدريب/إعداد البيانات). يضمن أن "النبضة رقم N" بكل الأقطاب المستخرَجة
    لنفس المريض تُشير فعلياً لنفس الدورة القلبية الفيزيولوجية.

    Returns: (beats, r_peak_locations)
    """
    clean = denoise(raw, fs, cutoff_hz=cutoff_hz)
    clean = remove_baseline(clean, degree=6)
    r_locs = external_r_locs if external_r_locs is not None else detect_r_peaks(clean, fs)
    beats = extract_beats(clean, r_locs, pre, post)
    return beats, r_locs

