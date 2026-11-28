# ReID (roadmap M3)

ReID в проекте — это дополнительная метрика `DistFeatureCos` для
сопоставления детекций с треками внутри одной камеры. Она не заменяет IoU и
по умолчанию выключена.

## Модели

| Имя | Файл | Тип | Вход | Выход |
|---|---|---|---|---|
| person | `data/reid/osnet_x0_25_msmt17_batch.onnx` | person ReID (MSMT17) | `128×256` | 512 |
| vehicle | `data/reid/res50_256x256.onnx` | vehicle ReID (ResNet50) | `256×256` | 2048 |

`osnet_x0_25_msmt17_batch.onnx` — это тот же OSNet, у которого исправлен
фиксированный `Reshape [1, -1]` и batch-измерение сделано динамическим.
Исходный `osnet_x0_25_msmt17.onnx` оставлен как fallback: он работает, но
`EmbeddingsCalculator` вынужден считать его по одному кропу.

Пересоздать dynamic-batch версию:

```bash
python tools/patch_reid_dynamic_batch.py \
    data/reid/osnet_x0_25_msmt17.onnx \
    data/reid/osnet_x0_25_msmt17_batch.onnx
```

Веса не меняются, патчится только `Reshape`/batch-размерность.

## Конфигурация

```ini
[tracking]
# IoU + ReID; order: Centers, Rects, Jaccard, Hist, FeatureCos, Mahalanobis, LocalCartesian
distance_weights = 0;0;0.5;0;0.5;0;0
dist_thresh = 0.8

[reid]
enable = 1
models = person;vehicle
backend = cuda_fp16        # cuda_fp16 | cuda | opencl | cpu
max_batch = 32
gallery_size = 15
quality_gate = 1
min_quality = 0.35
min_crop_side = 16
cpu_max_objects = 8        # CPU fallback budget
cpu_interval = 5           # CPU fallback: ReID every N-th frame
warmup = 1

[reid_person]
model = ../../data/reid/osnet_x0_25_msmt17_batch.onnx
input_width = 128
input_height = 256
mean = 0.485;0.456;0.406
std = 0.229;0.224;0.225
swap_rb = 1
types = person

[reid_vehicle]
model = ../../data/reid/res50_256x256.onnx
input_width = 256
input_height = 256
mean = 0.485;0.456;0.406
std = 0.229;0.224;0.225
swap_rb = 1
types = car;bus;truck;vehicle
```

`mean`/`std` должны совпадать с пайплайном, на котором обучалась модель.
Для OSNet это стандартная ImageNet-нормализация и RGB; для vehicle-модели
значения нужно подтвердить у экспортёра. `swap_rb = 1` означает
BGR (OpenCV) → RGB.

`embeddingParams` можно задавать и программно, как в
`example/examples.h`: `TrackerSettings::EmbeddingParams` содержит тот же
набор полей.

## Как это работает

1. `CTracker::CalcEmbeddins` находит модель по типу объекта.
2. Кропы группируются по `modelId`; для каждой группы выполняется один
   `EmbeddingsCalculator::Extract` (с чанкованием по `max_batch`).
3. Модель применяет `(x/255 - mean) / std`, L2-нормализует выход и
   возвращает `[N, D]`.
4. В трек добавляется эмбеддинг только при принятом назначении и хорошем
   quality score. Галерея ограничена `gallery_size`; `CalcCosine` сравнивает
   наблюдение с галереей и возвращает минимальное расстояние.
5. Эмбеддинги разных моделей не сравниваются: у `RegionEmbedding` есть
   `modelId`, а у трека — `m_embeddingModelId`.

## GPU/CPU

* GPU: `DNN_BACKEND_CUDA` + `DNN_TARGET_CUDA_FP16` (если OpenCV собран с
  CUDA). При инициализации выполняется прогрев для нескольких batch-размеров,
  чтобы первый кадр не платил за cuDNN autotune.
* CPU: тот же код, но с бюджетом. Если объектов больше `cpu_max_objects`,
  ReID для этой модели пропускается; `cpu_interval > 1` считает ReID раз в
  N кадров. Это осознанный fallback: vehicle-модель на CPU слишком тяжёлая
  для 30 объектов в реальном времени.
* Если батч не поддерживается моделью, `EmbeddingsCalculator` откатывается
  на по-кроповый инференс и пишет предупреждение один раз.

## Ориентировочные замеры

RTX 4070 Laptop, OpenCV 4.14 с CUDA/cuDNN, FP16, медиана после прогрева,
только инференс без детектора и трекера:

| Модель | batch=1 | batch=8 | batch=16 | batch=30 |
|---|---:|---:|---:|---:|
| vehicle `res50_256x256` | 3.6 мс | 5.3 мс | 12.7 мс | 24.1 мс |
| person OSNet (dynamic) | 11.8 мс | 11.7 мс | 11.8 мс | 11.7 мс |

Для 30 объектов vehicle на CPU: ~1.1 с/кадр (batch 30), поэтому CPU-режим
не рассчитан на 30 fps. Person OSNet на CPU заметно легче, но при 30
объектах всё равно требует бюджета.

В синтетическом smoke-тесте 30 объектов (15 person + 15 vehicle) на GPU
устойчивый кадр после прогрева — порядка 50 мс, первый кадр дороже из-за
инициализации; цифры зависят от детектора и загрузки GPU.

## A/B на MOT20 и VisDrone-MOT val

Одинаковый TensorRT-детектор и UniversalTracker; отличаются только веса
ReID/галерея. Время включает чтение кадров, детекцию, трекинг, ReID и запись
результата. Подробная таблица и интерпретация — в `README.md`, раздел
«ReID benchmark».

| Dataset | ReID | MOTA | IDF1 | IDSW | FP | FN | ms/frame |
|---|---|---:|---:|---:|---:|---:|---:|
| MOT20 | off | 0.181 | 0.266 | 4786 | 110005 | 814420 | 33.6 |
| MOT20 | 0.5/0.5, gallery 15 | 0.111 | 0.136 | 10329 | 50597 | 947346 | 78.4 |
| MOT20 | 0.8/0.2, gallery 5 | 0.188 | 0.274 | 4493 | 96435 | 820714 | 76.4 |
| VisDrone | off | -1.913 | 0.196 | 4758 | 288279 | 39376 | 57.8 |
| VisDrone | 0.5/0.5, gallery 15 | -0.583 | 0.300 | 2432 | 130698 | 47545 | 427.6 |

На MOT20 агрессивный ReID (`0.5/0.5`) ухудшает IDF1/IDSW; снижение веса до
`0.2` и галереи до 5 (`min_quality=0.5`) даёт небольшой выигрыш. На VisDrone
ReID сильно снижает FP/IDSW и повышает IDF1, но теряет recall и работает
примерно в 7 раз медленнее. Это означает, что текущий ReID — не «включай и
забыл»: его нужно калибровать под сцену, и по умолчанию он выключен.

## Ограничения

* ReID внутри одной камеры; межкамерная галерея не реализована.
* Один тип объекта — одна модель; несколько моделей на один тип не
  поддерживаются.
* Порог/вес `DistFeatureCos` общий для всех моделей. Если распределения
  person и vehicle сильно различаются, калибруйте вес/порог на валидации.
* Модели и их веса не входят в git автоматически (`*.onnx` в `.gitignore`);
  для воспроизводимости храните манифест с путями и контрольными суммами.
