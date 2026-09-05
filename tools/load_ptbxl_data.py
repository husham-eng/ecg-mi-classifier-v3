"""
load_ptbxl_data.py
=====================
يحمّل بيانات PTB-XL الخام (ملفات WFDB .dat/.hea) من ptbxl_selection/<فئة>/،
يستخرج الإشارة الخام لكل قطب مدعوم (LeadI, aVR, V2, V6)، يمرّرها بخط
المعالجة الحالي (تمرير عالٍ -> تصحيح محلي PR -> اكتشاف R -> تقطيع ->
قطع ديناميكي حسب RR)، ويُرجع كل النبضات مع (patient_id, label, lead,
beat, cutoff_idx) لاستخدامها بالتدريب والتقييم.

⚠️ إصلاح جوهري (جلسة تشخيص المحازاة): مواقع R تُكتشَف الآن عبر **إجماع
كل الأقطاب المتاحة بالتسجيل** (detect_r_consensus)، بدل الاعتماد على
Lead II وحده كمرجع — أثبت هذا تحسيناً بجودة المحازاة، خصوصاً بفئتي
الاحتشاء السفلي IL/IPL حيث Lead II نفسه (كونه قطباً سفلياً) قد يتشوّه
بنفس المرض المطلوب تمييزه، فيصبح مرجعاً غير موثوق لتلك الفئات تحديداً.

⚠️ عمود إضافي: "cutoff_idx" -- نقطة القطع الديناميكية الخاصة بكل نبضة
(راجع preprocessing.compute_dynamic_cutoff)، إجباري تمريره لاحقاً لـ
classifier.train_lead_model (كـnormal_cutoffs/pathological_cutoffs).
"""

from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ecg_pipeline.preprocessing import (process_raw_signal, denoise, remove_baseline_highpass,
                                         detect_r_peaks, detect_r_consensus)

# مطابقة اسم مجلد الفئة بـPTB-XL مع رمز الفئة المستخدم بالمشروع الحالي
CATEGORY_TO_LABEL = {
    "NORM": "Normal",
    "AMI": "A",
    "ASMI": "AS",
    "ILMI": "IL",
    "IPLMI": "IPL",
    # "IMI" (احتشاء سفلي بدون تحديد وحشي) غير مستخدَم بمخطط الفئات الحالي - يُتجاهَل
}

# مطابقة اسم القطب بالمشروع مع اسم القناة الفعلي بملفات WFDB
LEAD_NAME_MAP = {
    "LeadI": "I",
    "aVR": "AVR",
    "V2": "V2",
    "V6": "V6",
}

SUPPORTED_LEADS = list(LEAD_NAME_MAP.keys())


def load_metadata(database_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(database_csv)
    return df.set_index("ecg_id")


def extract_ecg_id(hea_path: Path) -> int:
    """يستخرج ecg_id من اسم الملف (بعد إعادة تسمية .dat/.hea لمطابقة محتوى الheader)."""
    m = re.match(r"0*(\d+)_hr", hea_path.stem)
    return int(m.group(1))


def load_all_beats(selection_dir: Path, database_csv: Path,
                    pre: int = 100, post: int = 300, verbose: bool = True) -> pd.DataFrame:
    """
    يرجع DataFrame بعمود لكل: patient_id, ecg_id, label, lead, beat (np.ndarray),
    cutoff_idx (int) لكل نبضة مكتشفة، لكل قطب مدعوم، لكل سجل ضمن الفئات
    المستخدمة حالياً.
    """
    meta = load_metadata(database_csv)
    rows = []
    skipped_categories = set()

    for cat_dir in sorted(selection_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        label = CATEGORY_TO_LABEL.get(cat_dir.name)
        if label is None:
            skipped_categories.add(cat_dir.name)
            continue

        hea_files = sorted(cat_dir.glob("*.hea"))
        for hea in hea_files:
            ecg_id = extract_ecg_id(hea)
            if ecg_id not in meta.index:
                continue
            patient_id = meta.loc[ecg_id, "patient_id"]

            try:
                record = wfdb.rdrecord(str(hea.with_suffix("")))
            except Exception as e:
                print(f"  ⚠️ تعذّرت قراءة {hea.name} (ملف تالف على الأرجح) — تم تجاوزه: {e}")
                continue
            fs = record.fs
            sig_names = record.sig_name

            # ✅ إجماع كل الأقطاب المتاحة بالتسجيل بدل الاعتماد على Lead II وحده
            # (يستخدم إزالة أساس بتمرير عالٍ للاتساق الكامل مع process_raw_signal)
            candidates_per_lead = {}
            for name in sig_names:
                sig = record.p_signal[:, sig_names.index(name)]
                clean = remove_baseline_highpass(denoise(sig, fs), fs)
                candidates_per_lead[name] = detect_r_peaks(clean, fs, polarity_robust=True)
            consensus_r_locs = detect_r_consensus(candidates_per_lead, len(sig_names),
                                                   cluster_window_samples=int(0.08 * fs))
            if len(consensus_r_locs) == 0:
                continue  # لم يتفق أي عدد كافٍ من الأقطاب على أي نبضة — تسجيل رديء الجودة، يُتجاهَل

            for lead, wfdb_name in LEAD_NAME_MAP.items():
                if wfdb_name not in sig_names:
                    continue
                raw = record.p_signal[:, sig_names.index(wfdb_name)]
                beats, r_locs, cutoffs = process_raw_signal(raw, fs, pre=pre, post=post,
                                                              external_r_locs=consensus_r_locs)
                for b, cutoff in zip(beats, cutoffs):
                    rows.append({
                        "patient_id": patient_id, "ecg_id": ecg_id,
                        "label": label, "lead": lead, "beat": b, "cutoff_idx": int(cutoff),
                    })

        if verbose:
            print(f"  فئة {cat_dir.name} ({label}): {len(hea_files)} سجل")

    if skipped_categories and verbose:
        print(f"  فئات تم تجاهلها (غير مستخدمة بمخطط الفئات الحالي): {sorted(skipped_categories)}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    selection_dir = Path("/home/claude/ptbxl_data/ptbxl_selection")
    database_csv = Path("/mnt/user-data/uploads/ptbxl_database.csv")
    df = load_all_beats(selection_dir, database_csv)
    print(f"\nإجمالي النبضات المستخرجة: {len(df)}")
    print(df.groupby(["lead", "label"])["patient_id"].nunique())
    df.to_pickle("/home/claude/ecg_app/ptbxl_beats.pkl")
    print("\n✅ حُفظت كل النبضات (مع cutoff_idx) بـ ptbxl_beats.pkl")
