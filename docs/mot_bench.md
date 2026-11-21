# MOT benchmark: сводные результаты

Сравнение конфигураций детектора/трекера на MOT20 (train) и
VisDrone-MOT2019 (val). Метрики считает `tools/mot_eval` (семантика
py-motmetrics, кросс-чек с py-motmetrics 1.4.0 пройден): CLEAR MOT
(MOTA/MOTP) и ID-меры (IDF1/IDP/IDR/IDSW). Дополнительно приводятся
`Recall = matches / GT` и `Precision = matches / predictions` (совпадение
рамок, не ID-метрики).

Все прогоны — на TensorRT, с единым трекером **ByteTrack** и
`match_type=1` (LAPJV). OCV DNN CUDA из сравнения исключён (медленный и
неточный).

Железо: RTX 4070 Laptop (8 ГБ), TensorRT 10.16.1.11, CUDA 13.2.

## 1. Методика и воспроизведение

| | MOT20 (train) | VisDrone-MOT2019 (val) |
|---|---|---|
| eval-классы | 1 (Pedestrian) | 1–10 |
| ignore (давят FP) | 2,7,8,9,10,11,12 | 0, 11 |
| выброшены | 3,4,5,6 | — |
| `min_area` | 0 | 800 |

`min_area` для VisDrone — практика против потока детекций объектов,
отсутствующих в MOT-разметке (мелкие/дальние). Важно: фильтр применяется
**только на этапе оценки** (`mot_eval --min_area`), а не в трекере.

```powershell
# MOT20 (baseline)
powershell -ExecutionPolicy Bypass -File data\mot_bench.ps1 -Dataset mot20 -OutDir <dir>

# VisDrone: основной прогон (YOLOv8l-visdrone, TensorRT FP16 — по умолчанию)
powershell -ExecutionPolicy Bypass -File data\mot_bench.ps1 -Dataset visdrone -OutDir <dir>

# VisDrone: WALDO30, прокси по пересекающимся классам
powershell -ExecutionPolicy Bypass -File data\mot_bench.ps1 -Dataset visdrone `
  -VisDroneIni data\settings_yolov8_waldo30_visdrone_classes.ini -VisDroneExample 3 -OutDir <dir>

# VisDrone: WALDO30, все классы (только скорость)
powershell -ExecutionPolicy Bypass -File data\mot_bench.ps1 -Dataset visdrone `
  -VisDroneIni data\settings_yolov8_waldo30.ini -VisDroneExample 3 -NoEval -OutDir <dir>
```

Параметры прогонов — единые для всех конфигураций:

| Прогон | Модель (вход) | Бэкенд / точность | conf | `tracker_type` | `match_type` |
|---|---|---|---|---|---|
| MOT20 baseline | YOLO11s COCO person, 736×1280 | TensorRT FP32 | 0.3 | 1 (ByteTrack) | 1 (LAPJV) |
| VisDrone YOLOv8l | YOLOv8l-visdrone, 960×1280 | TensorRT FP16 | 0.25 | 1 (ByteTrack) | 1 (LAPJV) |
| VisDrone WALDO30 (пересек. классы) | YOLOv8l-P2, 1024×1024 | TensorRT FP16 | 0.25 | 1 (ByteTrack) | 1 (LAPJV) |
| VisDrone WALDO30 (все классы) | YOLOv8l-P2, 1024×1024 | TensorRT FP16 | 0.5 | 1 (ByteTrack) | 1 (LAPJV) |

ByteTrack-пороги (явно в ini, одинаковые везде): `bytetrack_track_buffer=30`,
`bytetrack_track_thresh=0.5`, `bytetrack_high_thresh=0.5`,
`bytetrack_match_thresh=0.8`. Примечание: `match_type` ByteTrack игнорирует
(он только для UniversalTracker), но значение держится одинаковым для
единообразия.

## 2. Сводные таблицы

Жирным выделено **лучшее значение в столбце** (↑ — больше лучше, ↓ —
меньше лучше). `Predictions`/`GT objects` — информационные счётчики, не
выделяются. Агрегаты по суммарным FP/FN/IDSW/IDTP; MOTP — взвешенный по
matches; IDF1/IDP/IDR — по суммарным IDTP/IDFP/IDFN.

### 2.1 MOT20 (train)

Единственная конфигурация: YOLO11s (COCO, person), 736×1280, TensorRT
FP32, conf 0.3, ByteTrack. 4 последовательности, 8 931 кадр, 1 134 614
GT-объектов. Выделять нечего — прогон один.

| Метрика | Значение |
|---|---:|
| MOTA | 0.1100 |
| IDF1 | 0.1509 |
| IDP | 0.6574 |
| IDR | 0.0853 |
| Recall (matches / GT) | 0.1138 |
| Precision (matches / pred) | 0.8771 |
| MOTP (IoU) | 0.7615 |
| IDSW | 1243 |
| FP | 2970 |
| FN | 1005542 |
| Predictions | 147151 |

Тайминги по MOT20 не собираются (`mot_bench.ps1` пишет `*_timing.csv`
только для VisDrone); грубая оценка полного прогона — ~6 мин.

### 2.2 VisDrone-MOT2019 (val)

Все прогоны: 7 последовательностей, 2 846 кадров, 114 132 GT-объекта;
общая оценка — классы 1–10, ignore 0/11, `min_area=800`.

**Качество:**

| Прогон | MOTA ↑ | IDF1 ↑ | IDP ↑ | IDR ↑ | Recall ↑ | Precision ↑ | MOTP (IoU) ↑ | IDSW ↓ | FP ↓ | FN ↓ | Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8l-visdrone, TensorRT FP16 | **0.3576** | **0.4659** | **0.5140** | **0.4261** | **0.5483** | **0.6615** | 0.7471 | 1890 | **19877** | **51549** | 94604 |
| WALDO30, пересек. классы, conf 0.25 | 0.0856 | 0.3248 | 0.4704 | 0.2481 | 0.2985 | 0.5661 | **0.7678** | **700** | 23607 | 80059 | 60189 |

**Скорость** (полный прогон: чтение кадров + детекция + трекинг + запись
результата):

| Прогон | Время, с ↓ | ms/кадр ↓ | FPS ↑ |
|---|---:|---:|---:|
| YOLOv8l-visdrone, TensorRT FP16 | 175.80 | 61.77 | 16.19 |
| WALDO30, пересек. классы, conf 0.25 | **158.77** | **55.79** | **17.93** |
| WALDO30, все классы, conf 0.5 | 159.28 | 55.97 | 17.87 |

**Как читать.** На едином трекере YOLOv8l-visdrone — основная модель на
VisDrone: она выигрывает по MOTA/IDF1/IDP/IDR/Recall/Precision и по FP/FN.
Строка WALDO30 — **прокси** (другая таксономия: Person/LightVehicle/Truck/
Bus/Bike ↔ классы VisDrone; Tricycle/Awning-tricycle покрыты частично), а не
замена VisDrone-модели. Её превосходство только в MOTP и IDSW — следствие
консервативного детектора: в ~2 раза меньше recall (0.30 против 0.55),
меньше и в среднем крупнее матчей (выше IoU) и меньше детекций (меньше смен
ID).

## 3. Оценка качества тестирования

### 3.1 Замечания

1. **Параметры трекинга теперь едины.** Раньше WALDO30 гонялся на ByteTrack,
   а YOLOv8l/MOT20 — на UniversalTracker+Hungarian, что делало сравнение по
   IDSW/IDF1 неатрибутируемым. Исправлено: все прогоны — `tracker_type=1`
   (ByteTrack) + `match_type=1`, ByteTrack-пороги зафиксированы одинаковыми.
2. **ByteTrack-пороги vs conf детектора.** При `track_thresh=high_thresh=0.5`
   детекции с conf ∈ [0.25, 0.5) не открывают новые треки, а только доживают
   существующие. Эффект виден в цифрах: на VisDrone ByteTrack обрушил FP
   (240 578 → 19 877) и поднял MOTA до +0.36, но на MOT20 recall упал
   0.28 → 0.11 (низкоуверенные детекции толпы перестали открывать треки).
   Пороги — осознанный дефолт; тюнинг — отдельная итерация (см. раздел 4).
3. **`min_area=800` не влияет на трекер.** Фильтр срабатывает только в
   `mot_eval`, поэтому трекер ведёт и пишет все мелкие детекции: завышает
   `Predictions`, время и IDSW.
4. **MOTP — не мера локализации при сравнении моделей с разным числом
   матчей.** У WALDO30 MOTP (IoU) выше, потому что он матчит меньше и в
   среднем более крупные/чистые боксы — смещение выборки, а не точность рамок.
5. **Устаревший комментарий `MatchBipart`.** В большинстве `settings_*.ini`
   вне бенчмарка остаётся комментарий `MatchHungrian = 0 / MatchBipart = 1`,
   хотя энума `MatchBipart` больше нет (реально `MatchLAPJV = 1`). В
   бенчмарк-конфигах (`settings_mot20.ini`, `settings_visdrone_trt_fp16.ini`)
   исправлено.

### 3.2 Неполнота

* MOT20 прогнан в одной конфигурации — сравнивать на MOT20 нечего, это
  чистый baseline без точек сравнения.
* Нет HOTA/AssA: IDF1/IDSW есть, но оценка качества ассоциации неполна.
* Нет систематического среза по conf × min_area × NMS: выбран один
  operating point, «лучшая» конфигурация не подкреплена кривой
  Precision–Recall или HOTA.
* Нет таймингов по MOT20 (см. 2.1).

## 4. Что улучшить, чтобы поднять показатели

1. **MOT20 — поднять recall (главный ограничитель, сейчас ~0.11).** COCO-модель
   находит лишь ~11 % GT; ByteTrack с порогами 0.5 усугубил это (см. п.3).
   Точки роста: crowd-обученная модель (MOT20/CrowdHuman), yolo11m/x, conf
   ниже 0.3, SAHI-нарезка, и согласование `track_thresh`/`high_thresh` с conf.
2. **VisDrone — добить FP при сохранении recall.** ByteTrack уже срезал FP в
   ~12 раз (240 578 → 19 877), но остаются FP от неразмеченных мелких объектов:
   поднять conf с 0.25, поднять `min_area` и отфильтровать мелкие объекты
   **в самом трекере** (а не только в метриках), тюнинг NMS.
3. **Тюнинг ByteTrack-порогов.** `track_thresh=high_thresh=0.5` при conf
   0.25–0.3 «отрезает» низкоуверенные детекции от открытия новых треков (на
   MOT20 это −17 п.п. recall). Прогнать сетку `track_thresh`/`high_thresh`
   (например 0.25/0.4 или 0.3/0.5) и `match_thresh` (0.8 → 0.5–0.7 для мелких
   быстрых объектов VisDrone) и выбрать по HOTA.
4. **Добавить HOTA и проварьировать operating point.** Проварьировать
   conf × min_area × NMS и выбирать конфигурацию по кривой (PR/HOTA), а не
   по единственной точке.
