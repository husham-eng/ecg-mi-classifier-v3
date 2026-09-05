"""
retrain_final_models.py
==========================
يعيد تدريب النماذج النهائية (نموذج مستقل لكل قطب) على **كل** بيانات PTB-XL
المتاحة، باستخدام كل الإصلاحات المُثبَتة تجريبياً:
  - محازاة إجماع متعدد الأقطاب (بدل Lead II وحده) -- عبر load_ptbxl_data.py
  - كشف R مقاوم لانعكاس القطبية
  - إزالة أساس بتمرير عالٍ سريري العتبة (بدل polyfit عام يمتصّ ST حقيقي)
  - تصحيح محلي لكل نبضة بالنسبة لمقطع PR الخاص بها
  - نافذة قطع ديناميكية لكل نبضة حسب معدل نبضها الفعلي (عمود cutoff_idx)
  - بناء مرجع طبيعي مع استبعاد الشواذ (trim_fraction=0.1)
  - عرض مدى طبيعي مُحسَّن لكل قطب (k تكيّفي عبر LEAD_DEFAULT_K -- بدل
    القيمة القديمة الموحّدة 2.5)
  - Random Forest منظَّم (max_depth=12, min_samples_leaf=4)

هذا النموذج النهائي "للنشر" — يُدرَّب على كل البيانات المتاحة دفعة واحدة.
يحفظ النماذج في models/<lead>/.

⚠️ تحذير مهم: ملف `ptbxl_beats.pkl` لازم يكون منتجاً من نسخة
`load_ptbxl_data.py` **المُحدَّثة** (تحتوي عمود "cutoff_idx"). شغّل
load_ptbxl_data.py من جديد أولاً إذا كان الملف الحالي أقدم من هذا التحديث.
"""

from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ecg_pipeline.classifier import train_lead_model, MODELS_DIR

LEAD_CATEGORIES = {
    "LeadI": ["A", "AS", "IL", "IPL"],
    "aVR": ["AS", "IL", "IPL"],
    "V2": ["AS", "IL", "IPL"],
    "V6": ["AS", "IL", "IPL"],
}


def main():
    df = pd.read_pickle("/home/claude/ecg_app/ptbxl_beats.pkl")  # مُستخرَجة بمحازاة إجماع + cutoff_idx
    if "cutoff_idx" not in df.columns:
        raise SystemExit(
            "❌ ptbxl_beats.pkl لا يحتوي عمود cutoff_idx — شغّل load_ptbxl_data.py "
            "المُحدَّث من جديد أولاً قبل إعادة التدريب."
        )

    for lead, cats in LEAD_CATEGORIES.items():
        print(f"=== تدريب نموذج قطب {lead} (بيانات مُحاذاة، كل المرضى) ===")
        sub = df[df["lead"] == lead]

        normal_sub = sub[sub["label"] == "Normal"]
        normal_beats = np.stack(normal_sub["beat"].values)
        normal_cutoffs = normal_sub["cutoff_idx"].values

        pathological_beats = {c: np.stack(sub[sub["label"] == c]["beat"].values) for c in cats}
        pathological_cutoffs = {c: sub[sub["label"] == c]["cutoff_idx"].values for c in cats}

        model = train_lead_model(
            lead, normal_beats, pathological_beats,
            k_std=None,  # None => يستخدم تلقائياً LEAD_DEFAULT_K[lead] المُحسَّنة لكل قطب
            n_estimators=300,
            max_depth=12, min_samples_leaf=4, max_features="sqrt",
            normal_cutoffs=normal_cutoffs, pathological_cutoffs=pathological_cutoffs,
        )
        model.save(MODELS_DIR / lead)

        n_total = sum(len(b) for b in pathological_beats.values())
        n_patients = sub["patient_id"].nunique()
        print(f"  مرضى: {n_patients} | نبضات مرضية: {n_total} | نبضات طبيعية: {len(normal_beats)}")
        print(f"  الفئات: {model.classes}")
        depths = [e.get_depth() for e in model.clf.estimators_]
        print(f"  متوسط عمق الشجرة: {np.mean(depths):.1f} (كانت 69-122 قبل التنظيم)")
        print()

    print("✅ انتهى تدريب كل النماذج النهائية المُحدَّثة. الملفات محفوظة في:", MODELS_DIR)


if __name__ == "__main__":
    main()
