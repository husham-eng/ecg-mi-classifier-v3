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
    """⚠️ محفوظة للتوافق الخلفي فقط (استُخدمت بالجلسات السابقة). لا تُستخدم
    بعد الآن داخل process_raw_signal -- راجع remove_baseline_highpass أدناه
    وسبب الاستبدال (قيد موثَّق بجلسة تشخيص المحازاة/التقطيع)."""
    n = len(x)
    t = np.arange(1, n + 1, dtype=float)
    return x - np.polyval(np.polyfit(t, x, degree), t)


def remove_baseline_highpass(x: np.ndarray, fs: float, cutoff_hz: float = 0.5,
                              order: int = 4) -> np.ndarray:
    """
    ⚠️ إصلاح جوهري (جلسة تشخيص المحازاة/الأساس): يستبدل الـpolyfit العام
    (درجة 6 على كامل الشريط بعدة نبضات) بمرشح تمرير عالٍ (طور صفري،
    filtfilt) بعتبة 0.5Hz -- قريب من المعيار السريري الشائع (AAMI/ANSI
    EC57 يستخدم ~0.5-0.67Hz) لإزالة انجراف خط الأساس (تنفّس/حركة قطب)
    دون المساس بمحتوى ST-T الحقيقي.

    لماذا الاستبدال: منحنى بولينومي درجة 6 يُطبَّق على شريط كامل (~10
    ثوانٍ، عدة نبضات) لا يميّز بين "انجراف بطيء غير مرضي" وبين "ارتفاع/
    انخفاض ST حقيقي مستمر عبر الشريط" (وهو بالضبط ما يميّز فئات الاحتشاء
    المستهدفة) -- فقد يمتصّ جزءاً من الانحراف المرضي الحقيقي كأنه ضوضاء،
    بينما لا يوجد شيء مماثل ليُفقَد بفئة Normal أصلاً. الأثر العملي:
    معالجة "متطابقة بالكود" لكن غير متكافئة بالنتيجة بين الفئتين.
    مرشّح تمرير عالٍ بعتبة منخفضة ومضبوطة سريرياً أقل عرضة لهذا الالتباس.
    """
    from scipy.signal import butter
    nyq = fs / 2.0
    b, a = butter(order, cutoff_hz / nyq, btype="high")
    return filtfilt(b, a, x)


def local_isoelectric_correct(beat: np.ndarray, pre: int,
                               iso_start_samples: int = -50,
                               iso_end_samples: int = -20) -> np.ndarray:
    """
    ⚠️ إصلاح جوهري ثانٍ (جلسة تشخيص التقطيع/Stemming): يعيد تصفير كل
    نبضة -- طبيعية أو مرضية على حدٍّ سواء، بنفس الدالة ونفس المعاملات
    تماماً -- بالنسبة لخط تساوٍ كهربائي محلي خاص بها (تقدير لمقطع PR،
    بعد نهاية موجة P وقبل بداية QRS)، بدل ما تبقى مرتبطة بأي مرجع "صفر"
    مشترك موروث من معالجة الشريط الكامل.

    هذا يفكّ ارتباط أي مقارنة لاحقة (بناء المرجع الطبيعي + Stemming)
    عن افتراض أن الإشارة "كهربية" بجهد مطلق مشترك، ويجعلها فعلياً مقارنة
    (موضع زمني j، قيمة انحراف محلي v) -- أي بالضبط ما يفترضه المحتوى
    البحثي (معالجة مورفولوجية للشكل الهندسي، لا قيمة جهد مطلقة).

    iso_start_samples / iso_end_samples: حدود نافذة التساوي الكهربائي
    بالنسبة لقمة R (سالبة = قبل R؛ القيم الافتراضية -50 و-20 عينة تقابل
    -100ms إلى -40ms عند fs=500Hz). هذا تقدير عملي لمقطع PR بغياب
    خوارزمية ترسيم فعلية لبداية QRS بالمشروع حالياً (قيد موثَّق بالـ
    README قسم 4) -- قد يحتاج ضبطاً دقيقاً لاحقاً حسب معدل ضربات القلب
    الفعلي بالعيّنة.
    """
    lo, hi = pre + iso_start_samples, pre + iso_end_samples
    lo, hi = max(0, lo), min(len(beat), hi)
    if hi <= lo:
        return beat  # نافذة غير صالحة (نبضة قصيرة جداً بالنسبة للإزاحة) -- إرجاع كما هي بلا تصحيح
    iso_ref = np.median(beat[lo:hi])
    return beat - iso_ref


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


def compute_rr_to_next(r_locs: np.ndarray) -> np.ndarray:
    """المسافة (بالعينات) من كل قمة R إلى القمة التالية لها مباشرة؛
    NaN لآخر قمة بالتسجيل (لا معلومة تدل على وجود نبضة تالية)."""
    rr = np.full(len(r_locs), np.nan)
    if len(r_locs) > 1:
        rr[:-1] = np.diff(r_locs)
    return rr


def compute_dynamic_cutoff(rr_to_next: float, pre: int, post: int,
                            safety_margin_samples: int = 30) -> int:
    """
    ⚠️ إصلاح جوهري ثالث (جلسة تشخيص طول النافذة): نقطة القطع الخاصة
    بنبضة واحدة (index داخل مصفوفة النبضة، طولها pre+post) بدل حد ثابت
    للجميع. المعاينة البصرية (بمعية مستخدم المشروع) أثبتت أن نافذة
    خلفية ثابتة (post=300 عينة) تلامس أو تتخطى بداية النبضة التالية عند
    مرضى نبضهم أسرع من المتوسط -- 24.8% من النبضات فعلياً بعيّنة PTB-XL
    التجريبية كانت متأثرة. القطع بحد ثابت (جُرِّب عند 200 ثم 235) لم
    يحلّها جذرياً: بعض المرضى أسرع حتى من أي حد ثابت نختاره، فقفزة
    التلوّث تنتقل مكانها بدل ما تختفي (تأكَّد بصرياً بفئة A/LeadI تحديداً).

    الحل: نستخدم `rr_to_next` (ناتج compute_rr_to_next) لحساب نقطة قطع
    *خاصة بكل نبضة على حدة* حسب معدل نبضها الفعلي، مع هامش أمان قبل
    بداية QRS التالية (افتراضياً 30 عينة ≈ 60ms عند fs=500Hz).

    ⚠️ ملاحظة معمارية مهمة لأي نشر مستقبلي بالزمن الحقيقي (Real-time):
    هذا يتطلب معرفة موضع قمة R *التالية*، أي أن تصنيف نبضة واحدة لا
    يكتمل فعلياً إلا بعد وصول النبضة التي تليها (تأخير طبيعي ~ فترة RR
    واحدة). هذا غير مؤثر بالاستخدام الحالي للمشروع (تحليل دفعي لتسجيلات
    PTB-XL كاملة مسبقاً)، لكنه قيد يجب توثيقه لأي واجهة تصنيف لحظي لاحقاً؛
    عند غياب معرفة النبضة التالية (rr_to_next = NaN)، نستخدم النافذة
    الكاملة افتراضياً (لا دليل تلوّث، لا داعٍ للقطع).
    """
    length = pre + post
    if rr_to_next is None or (isinstance(rr_to_next, float) and np.isnan(rr_to_next)):
        return length
    valid_post = max(0, int(rr_to_next) - safety_margin_samples)
    return pre + min(post, valid_post)


def extract_beats(x: np.ndarray, r_locs: np.ndarray, pre: int, post: int) -> tuple[np.ndarray, np.ndarray]:
    """يقتطع نافذة ثابتة (pre عينة قبل R، post عينة بعده) حول كل قمة R.
    يرجع أيضاً *فهارس* r_locs الأصلية اللي فعلاً اتقطعت (بعض القمم قرب
    حواف التسجيل تُستبعَد لعدم توفّر عينات كافية) -- ضرورية لمطابقة كل
    نبضة مقطوعة بقيمة rr_to_next الصحيحة الخاصة بها لاحقاً."""
    beats, kept_idx = [], []
    n = len(x)
    for i, r in enumerate(r_locs):
        lo, hi = r - pre, r + post
        if lo >= 0 and hi <= n:
            beats.append(x[lo:hi])
            kept_idx.append(i)
    return np.array(beats), np.array(kept_idx, dtype=int)


def process_raw_signal(raw: np.ndarray, fs: float, pre: int = 200, post: int = 600,
                        cutoff_hz: float = 75.0,
                        external_r_locs: np.ndarray | None = None,
                        baseline_cutoff_hz: float = 0.5,
                        iso_window_samples: tuple[int, int] = (-50, -20),
                        safety_margin_samples: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    خط المعالجة الكامل من إشارة خام إلى نبضات مقطوعة ومحاذاة ومصحَّحة محلياً،
    مع نقطة قطع ديناميكية لكل نبضة (راجع compute_dynamic_cutoff).

    external_r_locs: مواقع R جاهزة (يُفضَّل أن تكون ناتج detect_r_consensus
    بدل قطب مرجعي واحد فقط، عند توفّر تسجيل متعدد الأقطاب كاملاً وقت
    التدريب/إعداد البيانات). يضمن أن "النبضة رقم N" بكل الأقطاب المستخرَجة
    لنفس المريض تُشير فعلياً لنفس الدورة القلبية الفيزيولوجية.

    ⚠️ ثلاثة إصلاحات جوهرية مدموجة هنا (تسري تلقائياً على كل مستدعي هذه
    الدالة -- تدريب واستدلال معاً، بلا حاجة لتعديل كل موضع استخدام على
    حدة):
      1) إزالة الأساس: مرشّح تمرير عالٍ سريري العتبة (remove_baseline_highpass)
         بدل polyfit عام على الشريط الكامل.
      2) تصحيح محلي لكل نبضة بالنسبة لمقطع PR الخاص بها (local_isoelectric_correct).
      3) نقطة قطع ديناميكية لكل نبضة حسب معدل نبضها الفعلي (compute_dynamic_cutoff)
         -- تُستخدَم لاحقاً بـreference.build_normal_envelope وreference.stem_beat
         لتجاهل ذيل أي نبضة يتلوّث ببداية النبضة التالية، بدل حد ثابت للجميع.

    بنفس المعاملات تماماً لأي فئة (طبيعي أو مرضي) ولأي قطب. راجع توثيق
    remove_baseline_highpass وlocal_isoelectric_correct وcompute_dynamic_cutoff أعلاه.

    Returns: (beats, r_peak_locations, cutoff_indices) -- beats مصحَّحة
    محلياً وجاهزة مباشرة لـreference.build_normal_envelope وreference.stem_beat؛
    cutoff_indices بنفس الطول والترتيب، تمرَّر لهما إجبارياً.
    """
    clean = denoise(raw, fs, cutoff_hz=cutoff_hz)
    clean = remove_baseline_highpass(clean, fs, cutoff_hz=baseline_cutoff_hz)
    r_locs = external_r_locs if external_r_locs is not None else detect_r_peaks(clean, fs)
    beats, kept_idx = extract_beats(clean, r_locs, pre, post)
    beats = np.array([
        local_isoelectric_correct(b, pre, iso_window_samples[0], iso_window_samples[1])
        for b in beats
    ])
    rr_to_next_all = compute_rr_to_next(r_locs)
    rr_to_next = rr_to_next_all[kept_idx]
    cutoffs = np.array([
        compute_dynamic_cutoff(rr, pre, post, safety_margin_samples) for rr in rr_to_next
    ], dtype=int)
    return beats, r_locs[kept_idx], cutoffs
