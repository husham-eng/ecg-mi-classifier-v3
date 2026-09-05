"""
translations.py
==================
نصوص واجهة المستخدم بثلاث لغات (عربي، إنجليزي، روسي). يُستخدَم من app.py
لتمرير القاموس المناسب للقالب حسب معامل الرابط ?lang=ar|en|ru.
"""

TRANSLATIONS = {
    "ar": {
        "dir": "rtl",
        "lang_code": "ar",
        "page_title": "تصنيف احتشاء عضلة القلب من ECG",
        "heading": "تصنيف أولي لنوع احتشاء عضلة القلب من ECG",
        "disclaimer": "⚠️ هذا نموذج بحثي أولي وليس جهازاً طبياً معتمداً. النتيجة للمساعدة على "
                      "تحديد الأولوية فقط، ويجب دائماً تأكيدها عبر تقييم طبي مختص.",
        "how_to_use_title": "📖 كيف تستخدم هذا التطبيق؟ (اضغط لعرض/إخفاء الشرح)",
        "how_to_use_html": """
            <p><strong>الفكرة الأساسية:</strong> التطبيق يقارن شكل نبضة القلب بكل قطب على حدة
            بمدى إحصائي "طبيعي"، ثم يدمج نتائج كل الأقطاب المرفوعة للوصول لتصنيف نهائي.</p>
            <p><strong>⚠️ مهم جداً:</strong> كل حقل قطب (Lead I, aVR, V2, V6) يحتاج
            <u>ملفاً أو صورة مختلفة وخاصة به فقط</u> — لا ترفع نفس الملف أو نفس صورة اللوحة
            الكاملة (12 قطباً) لكل الحقول. كل قطب يقيس إشارة كهربائية مختلفة فعلياً، ورفع نفس
            المحتوى لكل الحقول يعطي نتيجة خاطئة أو مرفوضة.</p>
            <p><strong>الملفات المقبولة لكل قطب:</strong></p>
            <ul>
              <li><b>ملف إشارة رقمية (CSV أو TXT)</b>: عمود واحد من الأرقام (قيم الجهد الكهربائي
                  بالتتابع الزمني)، مع تحديد معدل العيّنات (Hz) يدوياً.</li>
              <li><b>صورة (JPG/PNG)</b>: صورة لشريط قطب <u>واحد فقط</u> (خط إشارة واحد، ليس
                  لوحة كاملة فيها عدة أقطاب مرصوصة). التقطها بالكاميرا مباشرة أو ارفعها من المعرض.</li>
            </ul>
            <p><strong>حقل Lead II (اختياري، أعلى الصفحة):</strong> إذا توفر لديك تسجيل Lead II
            (حتى لو لم يكن من الأقطاب المصنَّفة)، ارفعه <u>كملف CSV/TXT فقط</u> (الصور غير
            مدعومة لهذا الحقل حالياً). يُستخدَم فقط لتحديد مواقع النبضات بدقة أعلى عبر كل
            الأقطاب الأخرى — تجربة فعلية أظهرت تحسناً ملموساً بالدقة (خصوصاً لقطب aVR).</p>
            <p><strong>خطوات الاستخدام:</strong></p>
            <ol>
              <li>اختر قطباً واحداً على الأقل من الأقطاب الأربعة أدناه.</li>
              <li>ارفع له ملف إشارة أو صورة <u>خاصة بهذا القطب تحديداً</u>.</li>
              <li>حدّد معدل العيّنات (Hz) إن كان الملف CSV/TXT (يُكتشف تلقائياً للصور).</li>
              <li>كرّر لبقية الأقطاب المتوفرة لديك (اختياري، لكن كل ما زاد العدد زادت الدقة).</li>
              <li>اضغط زر "تصنيف" وانتظر النتيجة.</li>
            </ol>
        """,
        "lead_ii_title": "Lead II (اختياري — لتحسين دقة المحازاة فقط)",
        "lead_ii_desc": "لو توفر لديك تخطيط Lead II الكامل (حتى لو لم يكن مطلوباً بالتصنيف)، "
                        "ارفعه هنا كملف CSV/TXT فقط: يُستخدَم فقط لتحديد مواقع نبضات القلب "
                        "بدقة أعلى عبر كل الأقطاب المرفوعة أدناه.",
        "lead_ii_file_label": "ملف إشارة Lead II (CSV/TXT فقط):",
        "fs_label": "معدل العينات Hz:",
        "doctor_email_label": "البريد الإلكتروني لإرسال التقرير (اختياري):",
        "doctor_email_hint": "إن أدخلته، سيُرسَل تقرير مفصَّل (أشكال + جداول احتمالات) فور اكتمال التصنيف — لمراجعة الطبيب/الفني بدراسة التحقق الميداني.",
        "intro_text": "ارفع ملف إشارة (CSV/TXT) أو صورة مخطط (JPG/PNG) لقطب واحد أو أكثر من "
                      "الأقطاب المدعومة أدناه. <strong>كل قطب يحتاج ملفه/صورته الخاصة والمختلفة "
                      "عن غيره.</strong> كل ما زاد عدد الأقطاب المرفوعة، كل ما كان القرار "
                      "النهائي أدق (يُدمج تلقائياً بتصويت مرجّح).",
        "lead_box_title": "قطب {lead}",
        "file_label": "ملف (اختياري):",
        "fs_hint": "للصور: يُكتشف معدل العينات تلقائياً من تحليل الصورة.",
        "submit_button": "تصنيف",
        "processing": "جاري المعالجة...",
        "error_prefix": "خطأ: ",
        "lang_selector_label": "اللغة:",
        "panel_section_title": "🚀 الطريقة السريعة: رفع صورة اللوحة الكاملة (12 قطباً)",
        "panel_section_desc": "صوّر أو ارفع صورة تخطيط ECG الكاملة (كل الأقطاب الـ12 بلوحة واحدة)، "
                               "والنظام يكتشف ويقصّ الأقطاب المطلوبة تلقائياً. "
                               "<strong>هذي ميزة تجريبية</strong> — تأكّد دائماً من صحة الأقطاب "
                               "المكتشفة بالمعاينة أدناه قبل التصنيف.",
        "panel_file_label": "صورة اللوحة الكاملة:",
        "panel_detect_button": "اكتشاف الأقطاب",
        "panel_detecting": "جاري تحليل الصورة...",
        "panel_detect_error_prefix": "تعذّر الاكتشاف: ",
        "confidence_confirmed": "✅ مؤكَّد (قراءة نصية واضحة)",
        "confidence_weak": "⚠️ غير مؤكَّد — تحقق من الصورة",
        "confidence_not_found": "⚠️ لم تُقرأ التسمية — تحقق من الصورة",
        "panel_confirm_button": "تأكيد الأقطاب وتصنيف",
        "panel_review_note": "راجع كل قطب مكتشَف أدناه. لو أي واحد يبدو خاطئاً، تجاهل هذا القسم "
                              "واستخدم الرفع اليدوي بالأسفل بدلاً منه لهذا القطب تحديداً.",
        "manual_section_divider": "— أو: الطريقة اليدوية (رفع كل قطب على حدة) —",
    },
    "en": {
        "dir": "ltr",
        "lang_code": "en",
        "page_title": "ECG Myocardial Infarction Classifier",
        "heading": "Preliminary MI Type Classification from ECG",
        "disclaimer": "⚠️ This is a preliminary research prototype, not an approved medical "
                      "device. Results are for triage assistance only and must always be "
                      "confirmed by a qualified medical evaluation.",
        "how_to_use_title": "📖 How to use this app? (click to show/hide)",
        "how_to_use_html": """
            <p><strong>Core idea:</strong> the app compares each lead's beat shape against a
            statistical "normal" range, then combines the results from all uploaded leads
            into one final classification.</p>
            <p><strong>⚠️ Very important:</strong> each lead field (Lead I, aVR, V2, V6) needs
            <u>its own distinct file or image</u> — do NOT upload the same file, or the same
            full 12-lead panel image, to every field. Each lead measures a genuinely different
            electrical signal; uploading identical content to all fields will produce a wrong
            or rejected result.</p>
            <p><strong>Accepted files per lead:</strong></p>
            <ul>
              <li><b>Digital signal file (CSV or TXT)</b>: a single column of numbers (voltage
                  values over time), with the sampling rate (Hz) set manually.</li>
              <li><b>Image (JPG/PNG)</b>: a photo of a <u>single lead strip only</u> (one trace,
                  not a full panel with several leads stacked together). Take it directly with
                  your camera or upload from your gallery.</li>
            </ul>
            <p><strong>Lead II field (optional, top of page):</strong> if you have a Lead II
            recording available (even if Lead II itself isn't one of the classified leads),
            upload it <u>as a CSV/TXT file only</u> (images aren't supported for this field
            yet). It is used only to pinpoint beat locations more accurately across all other
            leads — real testing showed a measurable accuracy improvement, especially for aVR.</p>
            <p><strong>Steps to use:</strong></p>
            <ol>
              <li>Pick at least one of the four leads below.</li>
              <li>Upload a signal file or image <u>specific to that exact lead</u>.</li>
              <li>Set the sampling rate (Hz) for CSV/TXT files (auto-detected for images).</li>
              <li>Repeat for any other leads you have available (optional, but more leads
                  means better accuracy).</li>
              <li>Click "Classify" and wait for the result.</li>
            </ol>
        """,
        "lead_ii_title": "Lead II (optional — improves alignment accuracy only)",
        "lead_ii_desc": "If you have a full Lead II recording available (even if not required "
                        "for classification), upload it here as a CSV/TXT file only: it is "
                        "used solely to pinpoint heartbeat locations more accurately across "
                        "all leads uploaded below.",
        "lead_ii_file_label": "Lead II signal file (CSV/TXT only):",
        "fs_label": "Sampling rate (Hz):",
        "doctor_email_label": "Email address for report (optional):",
        "doctor_email_hint": "If provided, a detailed report (charts + probability tables) will be emailed immediately after classification — for field-study / clinician review.",
        "intro_text": "Upload a signal file (CSV/TXT) or a strip image (JPG/PNG) for one or "
                      "more of the supported leads below. <strong>Each lead needs its own, "
                      "distinct file/image.</strong> The more leads you upload, the more "
                      "accurate the final decision (automatically combined via weighted "
                      "voting).",
        "lead_box_title": "Lead {lead}",
        "file_label": "File (optional):",
        "fs_hint": "For images: the sampling rate is auto-detected from image analysis.",
        "submit_button": "Classify",
        "processing": "Processing...",
        "error_prefix": "Error: ",
        "lang_selector_label": "Language:",
        "panel_section_title": "🚀 Fast method: upload the full 12-lead panel image",
        "panel_section_desc": "Photograph or upload the complete ECG panel (all 12 leads in one "
                               "sheet), and the system will automatically detect and crop the "
                               "leads it needs. <strong>This is an experimental feature</strong> "
                               "— always verify the detected leads in the preview below before "
                               "classifying.",
        "panel_file_label": "Full panel image:",
        "panel_detect_button": "Detect leads",
        "panel_detecting": "Analyzing image...",
        "panel_detect_error_prefix": "Detection failed: ",
        "confidence_confirmed": "✅ Confirmed (clear text reading)",
        "confidence_weak": "⚠️ Not confirmed — please verify",
        "confidence_not_found": "⚠️ Label not readable — please verify",
        "panel_confirm_button": "Confirm leads and classify",
        "panel_review_note": "Review each detected lead below. If any looks wrong, ignore this "
                              "section and use the manual upload below instead for that "
                              "specific lead.",
        "manual_section_divider": "— or: Manual method (upload each lead separately) —",
    },
    "ru": {
        "dir": "ltr",
        "lang_code": "ru",
        "page_title": "Классификатор инфаркта миокарда по ЭКГ",
        "heading": "Предварительная классификация типа ИМ по ЭКГ",
        "disclaimer": "⚠️ Это предварительный исследовательский прототип, а не "
                      "сертифицированное медицинское устройство. Результат предназначен "
                      "только для помощи в определении приоритетности и всегда должен быть "
                      "подтверждён квалифицированной медицинской оценкой.",
        "how_to_use_title": "📖 Как пользоваться приложением? (нажмите, чтобы показать/скрыть)",
        "how_to_use_html": """
            <p><strong>Основная идея:</strong> приложение сравнивает форму сердечного цикла
            по каждому отведению со статистической "нормой", затем объединяет результаты всех
            загруженных отведений в один итоговый диагноз.</p>
            <p><strong>⚠️ Очень важно:</strong> каждому полю отведения (Lead I, aVR, V2, V6)
            нужен <u>свой собственный, отдельный файл или изображение</u> — не загружайте один
            и тот же файл или одно изображение полной панели из 12 отведений во все поля.
            Каждое отведение измеряет действительно разный электрический сигнал; загрузка
            одинакового содержимого во все поля приведёт к неверному или отклонённому
            результату.</p>
            <p><strong>Допустимые файлы для каждого отведения:</strong></p>
            <ul>
              <li><b>Файл цифрового сигнала (CSV или TXT)</b>: один столбец чисел (значения
                  напряжения во времени), с указанием частоты дискретизации (Гц) вручную.</li>
              <li><b>Изображение (JPG/PNG)</b>: фото <u>только одной полосы отведения</u>
                  (одна кривая, а не полная панель с несколькими отведениями). Снимите камерой
                  напрямую или загрузите из галереи.</li>
            </ul>
            <p><strong>Поле Lead II (необязательно, вверху страницы):</strong> если у вас есть
            запись Lead II (даже если само Lead II не входит в классифицируемые отведения),
            загрузите её <u>только как файл CSV/TXT</u> (изображения для этого поля пока не
            поддерживаются). Используется только для более точного определения положения
            сердечных циклов по всем остальным отведениям — реальное тестирование показало
            заметное улучшение точности, особенно для aVR.</p>
            <p><strong>Шаги использования:</strong></p>
            <ol>
              <li>Выберите хотя бы одно из четырёх отведений ниже.</li>
              <li>Загрузите файл сигнала или изображение <u>именно для этого отведения</u>.</li>
              <li>Укажите частоту дискретизации (Гц) для файлов CSV/TXT (для изображений
                  определяется автоматически).</li>
              <li>Повторите для остальных доступных вам отведений (необязательно, но больше
                  отведений — выше точность).</li>
              <li>Нажмите «Классифицировать» и дождитесь результата.</li>
            </ol>
        """,
        "lead_ii_title": "Lead II (необязательно — только для повышения точности выравнивания)",
        "lead_ii_desc": "Если у вас есть полная запись Lead II (даже если она не требуется для "
                        "классификации), загрузите её здесь только как файл CSV/TXT: она "
                        "используется исключительно для более точного определения положения "
                        "сердечных циклов по всем отведениям, загруженным ниже.",
        "lead_ii_file_label": "Файл сигнала Lead II (только CSV/TXT):",
        "fs_label": "Частота дискретизации (Гц):",
        "doctor_email_label": "Email для отчёта (необязательно):",
        "doctor_email_hint": "Если указан, подробный отчёт (графики + таблицы вероятностей) будет отправлен по email сразу после классификации — для проверки врачом.",
        "intro_text": "Загрузите файл сигнала (CSV/TXT) или изображение полосы (JPG/PNG) для "
                      "одного или нескольких поддерживаемых отведений ниже. <strong>Каждому "
                      "отведению нужен свой собственный, отдельный файл/изображение.</strong> "
                      "Чем больше отведений вы загрузите, тем точнее итоговое решение "
                      "(автоматически объединяется взвешенным голосованием).",
        "lead_box_title": "Отведение {lead}",
        "file_label": "Файл (необязательно):",
        "fs_hint": "Для изображений: частота дискретизации определяется автоматически.",
        "submit_button": "Классифицировать",
        "processing": "Обработка...",
        "error_prefix": "Ошибка: ",
        "lang_selector_label": "Язык:",
        "panel_section_title": "🚀 Быстрый способ: загрузите изображение полной панели (12 отведений)",
        "panel_section_desc": "Сфотографируйте или загрузите полное изображение ЭКГ (все 12 "
                               "отведений на одном листе), и система автоматически определит и "
                               "вырежет нужные отведения. <strong>Это экспериментальная "
                               "функция</strong> — всегда проверяйте обнаруженные отведения в "
                               "предпросмотре ниже перед классификацией.",
        "panel_file_label": "Изображение полной панели:",
        "panel_detect_button": "Определить отведения",
        "panel_detecting": "Анализ изображения...",
        "panel_detect_error_prefix": "Не удалось определить: ",
        "confidence_confirmed": "✅ Подтверждено (чёткое чтение текста)",
        "confidence_weak": "⚠️ Не подтверждено — проверьте вручную",
        "confidence_not_found": "⚠️ Метка не читается — проверьте вручную",
        "panel_confirm_button": "Подтвердить отведения и классифицировать",
        "panel_review_note": "Проверьте каждое обнаруженное отведение ниже. Если что-то выглядит "
                              "неверно, проигнорируйте этот раздел и используйте ручную загрузку "
                              "ниже для этого конкретного отведения.",
        "manual_section_divider": "— или: Ручной способ (загрузка каждого отведения отдельно) —",
    },
}

DEFAULT_LANG = "ar"


def get_translation(lang: str) -> dict:
    return TRANSLATIONS.get(lang, TRANSLATIONS[DEFAULT_LANG])
