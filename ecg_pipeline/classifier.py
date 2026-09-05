
"""
ecg_pipeline.classifier
==========================
تدريب وحفظ وتحميل نماذج التصنيف (نموذج مستقل لكل قطب مدعوم)، بالإضافة
لدالة التنبؤ الموحّدة على مستوى مريض واحد (نبضة واحدة أو أكثر).
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .reference import build_normal_envelope, stem_beat, is_likely_normal
from .weighting import weight_with_class_specificity, weight_single_beat, build_weight_lookup

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# ⚠️ قيم k مُحسَّنة تجريبياً لكل قطب (مسح شبكي + GroupKFold صارم على
# مستوى المريض على بيانات PTB-XL، موثَّق بمقالة المشروع). القيمة
# القديمة الموحّدة (k=2.5) كانت السبب الرئيسي وراء تصنيف أكثر من 70%
# من النبضات المرضية كـ"طبيعي" تلقائياً (المدى كان واسعاً جداً، فيبتلع
# أغلب الانحرافات المرضية قبل ما تصل خطوة التصنيف أصلاً). خفض k ضيّق
# المدى الطبيعي وسمح لمزيد من الانحرافات المرضية الحقيقية بالظهور،
# لكن بدون إفراط (القيم المثلى تقع فعلياً بين 1.25-1.5، لا أقل ولا أكثر
# — قيم أصغر تبدأ تُدخل تباين النبضات الطبيعية نفسها كـ"انحراف" خاطئ).
LEAD_DEFAULT_K = {
    "LeadI": 1.25,
    "aVR": 1.5,
    "V2": 1.25,
    "V6": 1.25,
}
DEFAULT_K_FALLBACK = 1.3  # لأي قطب غير مذكور أعلاه (احتياطي معقول قريب من كل القيم المثلى)


class LeadModel:
    """يجمع كل ما يلزم للتصنيف على قطب واحد: المرجع الطبيعي + النموذج + جداول الترجيح."""

    def __init__(self, lead: str, classes: list[str], window_len: int,
                 ref_min: np.ndarray, ref_max: np.ndarray,
                 idf_table: dict, specificity_table: dict,
                 clf: RandomForestClassifier):
        self.lead = lead
        self.classes = classes
        self.window_len = window_len
        self.ref_min = ref_min
        self.ref_max = ref_max
        self.idf_table = idf_table                # {(j, v): weight}
        self.specificity_table = specificity_table  # {(j, v): weight}
        self.clf = clf
        self._idf_lookup = build_weight_lookup(idf_table)
        self._specificity_lookup = build_weight_lookup(specificity_table)

    def predict_beat(self, beat: np.ndarray, normal_threshold: int = 50,
                      cutoff: int | None = None) -> dict:
        """
        cutoff: نقطة القطع الديناميكية الخاصة بهذي النبضة (ناتج
        preprocessing.compute_dynamic_cutoff، مبنية على مسافتها الفعلية
        لقمة R التالية). اتركه None (الافتراضي) لسلوك قديم مطابق تماماً
        (نافذة كاملة، بلا قطع) -- مناسب لأي استدعاء لا يعرف بعد أين تقع
        النبضة التالية (مثلاً نبضة تصل بزمن حقيقي). راجع ملاحظة القيد
        المعماري بتوثيق preprocessing.compute_dynamic_cutoff.
        """
        beat = np.asarray(beat, dtype=float)
        original_len = len(beat)
        if original_len != self.window_len:
            beat = np.interp(np.linspace(0, 1, self.window_len),
                              np.linspace(0, 1, original_len), beat)
            if cutoff is not None:
                cutoff = int(round(cutoff * self.window_len / original_len))

        stemmed = stem_beat(beat, self.ref_min, self.ref_max, cutoff=cutoff)

        if is_likely_normal(stemmed, threshold=normal_threshold):
            probs = {c: 0.0 for c in self.classes}
            probs["Normal"] = 1.0
            return probs

        weighted = weight_single_beat(stemmed, self._idf_lookup, self._specificity_lookup)
        proba = self.clf.predict_proba([weighted])[0]
        probs = dict(zip(self.clf.classes_, proba))
        probs.setdefault("Normal", 0.0)
        return probs

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self.clf, f)
        np.save(path / "ref_min.npy", self.ref_min)
        np.save(path / "ref_max.npy", self.ref_max)
        meta = {
            "lead": self.lead, "classes": self.classes, "window_len": self.window_len,
            "idf_table": _table_to_json(self.idf_table),
            "specificity_table": _table_to_json(self.specificity_table),
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: Path) -> "LeadModel":
        with open(path / "model.pkl", "rb") as f:
            clf = pickle.load(f)
        ref_min = np.load(path / "ref_min.npy")
        ref_max = np.load(path / "ref_max.npy")
        with open(path / "meta.json") as f:
            meta = json.load(f)
        idf_table = _table_from_json(meta["idf_table"])
        specificity_table = _table_from_json(meta["specificity_table"])
        return cls(meta["lead"], meta["classes"], meta["window_len"],
                    ref_min, ref_max, idf_table, specificity_table, clf)


def _table_to_json(table: dict) -> dict:
    """يحوّل {(j, v): وزن} إلى {"j:v": وزن} لأن مفاتيح JSON يجب أن تكون نصوصاً."""
    return {f"{j}:{v}": w for (j, v), w in table.items()}


def _table_from_json(obj: dict) -> dict:
    out = {}
    for k, w in obj.items():
        j_str, v_str = k.split(":", 1)
        out[(int(j_str), float(v_str))] = w
    return out


def _compute_idf_specificity(stemmed_beats: np.ndarray, labels: np.ndarray, decimals: int = 1):
    """
    يستخرج جداول IDF/الاختصاص من بيانات التدريب.
    ⚠️ المفتاح (الموضع j، القيمة v) بدل القيمة v وحدها — راجع weighting.py.
    """
    n_beats = stemmed_beats.shape[0]
    n_positions = stemmed_beats.shape[1]
    bins = np.round(stemmed_beats, decimals)
    classes = sorted(set(labels))
    class_beat_counts = {c: (labels == c).sum() for c in classes}

    from collections import defaultdict
    value_freq_per_class = defaultdict(lambda: defaultdict(int))
    doc_freq_total = defaultdict(int)

    for i, row in enumerate(bins):
        cls = labels[i]
        for j in range(n_positions):
            v = row[j]
            if v == 0:
                continue
            key = (j, float(v))
            value_freq_per_class[key][cls] += 1
            doc_freq_total[key] += 1

    specificity, idf = {}, {}
    for key, per_class in value_freq_per_class.items():
        freqs = np.array([per_class.get(c, 0) / class_beat_counts[c] for c in classes])
        specificity[key] = float(freqs.max() / (freqs.sum() + 1e-9))
        idf[key] = float(np.log(n_beats / doc_freq_total[key]))
    return idf, specificity


def train_lead_model(lead: str, normal_beats: np.ndarray,
                      pathological_beats: dict[str, np.ndarray],
                      k_std: float | None = None, n_estimators: int = 300,
                      max_depth: int | None = 12, min_samples_leaf: int = 4,
                      max_features: str | float = "sqrt",
                      normal_cutoffs: np.ndarray | None = None,
                      pathological_cutoffs: dict[str, np.ndarray] | None = None) -> LeadModel:
    """
    يدرّب نموذج قطب واحد كامل من الصفر: بناء مرجع طبيعي، Stemming،
    حساب أوزان IDF/الاختصاص، ثم تدريب Random Forest.

    k_std: عرض المدى الطبيعي (المتوسط ± k×الانحراف المعياري). إذا تُرك
    None (الافتراضي)، يُستخدَم تلقائياً أفضل قيمة مُحسَّنة لهذا القطب
    تحديداً من LEAD_DEFAULT_K (أو DEFAULT_K_FALLBACK إذا كان القطب غير
    مذكور) — راجع تعليق LEAD_DEFAULT_K أعلاه لتفاصيل كيفية استخراج هذي
    القيم ولماذا القيمة القديمة (2.5) كانت أوسع من اللازم بشكل كبير.

    normal_cutoffs / pathological_cutoffs: نقاط القطع الديناميكية (ناتج
    preprocessing.compute_dynamic_cutoff) بنفس شكل normal_beats/
    pathological_beats على التوالي (pathological_cutoffs قاموس بنفس
    مفاتيح pathological_beats). اتركهما None (الافتراضي) لتعطيل القطع
    الديناميكي كلياً -- تدريب مطابق تماماً للسلوك القديم بلا أي تغيير.
    """
    if k_std is None:
        k_std = LEAD_DEFAULT_K.get(lead, DEFAULT_K_FALLBACK)

    window_len = normal_beats.shape[1]
    ref_min, ref_max = build_normal_envelope(normal_beats, k=k_std, cutoffs=normal_cutoffs)

    X, y = [], []
    for cls, beats in pathological_beats.items():
        cutoffs_for_cls = None
        if pathological_cutoffs is not None:
            cutoffs_for_cls = pathological_cutoffs.get(cls)
        for i, b in enumerate(beats):
            c = cutoffs_for_cls[i] if cutoffs_for_cls is not None else None
            X.append(stem_beat(b, ref_min, ref_max, cutoff=c))
            y.append(cls)
    X = np.array(X)
    y = np.array(y)

    idf_table, specificity_table = _compute_idf_specificity(X, y)
    X_weighted = weight_with_class_specificity(X, y)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_weighted, y)

    classes = sorted(set(y)) + ["Normal"]
    return LeadModel(lead, classes, window_len, ref_min, ref_max, idf_table, specificity_table, clf)

