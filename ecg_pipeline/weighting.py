
"""
ecg_pipeline.weighting
=========================
تطبيع وترجيح السمات (TF-IDF + قمع التداخل بين الفئات / Class Specificity)
المُطبَّق على النبضات بعد Stemming.

⚠️ إصلاح جوهري (موثَّق بتفصيل بمقالة المشروع، قسم "منهجية التصحيح"):
النسخة السابقة كانت تحسب الوزن بالاعتماد على *قيمة الانحراف فقط*
(value)، متجاهلة تماماً *موضعها* (j) داخل النبضة — أي جزء من دورة
القلب حدث فيه هذا الانحراف (QRS مقابل ST مقابل موجة T). هذا كان يُسوّي
كل الإشارة الزمنية الغنية إلى دالة أحادية البعد للقيمة فقط، ويطمس
بالضبط المعلومة المكانية-الزمنية المميّزة لأنواع الاحتشاء المختلفة.

الإصلاح: هوية "السمة" أصبحت الزوج (موضع j، قيمة v) بدل القيمة v وحدها.
النتيجة: جدول ترجيح لكل موضع على حدة، بدل جدول عام واحد للنبضة كاملة.
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict


def weight_with_class_specificity(stemmed_beats: np.ndarray, labels: np.ndarray,
                                   decimals: int = 1) -> np.ndarray:
    """
    نفس فكرة IDF + الاختصاص الطبقي بالنسخة القديمة، لكن محسوبة **لكل موضع
    على حدة** بدل تجميع كل مواضع النبضة معاً.
    """
    n_beats = stemmed_beats.shape[0]
    n_positions = stemmed_beats.shape[1]
    bins = np.round(stemmed_beats, decimals)
    classes = sorted(set(labels))
    class_beat_counts = {c: (labels == c).sum() for c in classes}

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
        specificity[key] = freqs.max() / (freqs.sum() + 1e-9)
        idf[key] = np.log(n_beats / doc_freq_total[key])

    weighted = np.zeros_like(stemmed_beats)
    for i, row in enumerate(bins):
        for j in range(n_positions):
            v = row[j]
            if v == 0:
                continue
            key = (j, float(v))
            weighted[i, j] = stemmed_beats[i, j] * idf.get(key, 0.0) * specificity.get(key, 0.0)
    return weighted


def build_weight_lookup(table: dict) -> dict:
    """
    يحوّل جدول ترجيح {(j, v): وزن} إلى قاموس لكل موضع على حدة:
        {j: (مصفوفة قيم مرتّبة, مصفوفة أوزان مقابلة)}
    جاهز للاستيفاء الخطي **داخل نفس الموضع فقط** — لا نستوفي أبداً بين
    مواضع مختلفة، لأن قيمة انحراف عند موضع QRS لا معنى لمقارنتها بموضع
    الموجة T حتى لو تصادف تساويهما رقمياً.
    """
    per_position = defaultdict(dict)
    for key, weight in table.items():
        j, v = key
        per_position[j][v] = weight

    lookup = {}
    for j, value_weight_map in per_position.items():
        keys = np.array(sorted(value_weight_map.keys()), dtype=float)
        values = np.array([value_weight_map[k] for k in keys], dtype=float)
        lookup[j] = (keys, values)
    return lookup


def lookup_weight(position: int, v: float, lookup: dict) -> float:
    """
    يبحث عن وزن القيمة v عند الموضع المحدد، بالاستيفاء الخطي بين أقرب
    قيمتين *شوهدتا عند نفس الموضع بالتدريب*. لو الموضع نفسه لم يُشاهَد
    نشطاً إطلاقاً بالتدريب، نُرجع 0.0 بدل استيفاء مضلِّل من موضع مختلف.
    """
    if position not in lookup:
        return 0.0
    keys, values = lookup[position]
    if len(keys) == 0:
        return 0.0
    if len(keys) == 1:
        return float(values[0])
    return float(np.interp(v, keys, values))


def weight_single_beat(stemmed_beat: np.ndarray, idf_lookup: dict, specificity_lookup: dict,
                        decimals: int = 1) -> np.ndarray:
    """
    يطبّق أوزان IDF/الاختصاص المحسوبة مسبقاً (واعية بالموضع) على نبضة
    واحدة جديدة وقت الاستدلال.

    idf_lookup / specificity_lookup: يجب أن يكونا ناتج build_weight_lookup
    الجاهز مسبقاً — ابنِهما مرة واحدة عند تحميل/تدريب النموذج (كما تفعل
    LeadModel.__init__)، لا بكل استدعاء، لأسباب أداء.
    """
    bins = np.round(stemmed_beat, decimals)
    weighted = np.zeros_like(stemmed_beat)
    for j, v in enumerate(bins):
        if v != 0:
            weighted[j] = (stemmed_beat[j]
                           * lookup_weight(j, v, idf_lookup)
                           * lookup_weight(j, v, specificity_lookup))
    return weighted

