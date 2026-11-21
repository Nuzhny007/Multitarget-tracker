# GPU-пайплайн детекции для TensorRT (NVDEC → CUDA-препроцесс → TensorRT)

> **Дата:** 2026-11-21.
> **Статус:** анализ + план (код не менялся). Соответствует пункту roadmap M3
> «GPU-декодирование/препроцессинг (опционально)».
> **Метод:** разбор исходного кода `Multitarget-tracker` и эталонной реализации
> `D-FINE-seg/cpp` (C:\work\yolo\D-FINE-seg\cpp).

---

## 1. Текущее состояние

Путь TensorRT-детектора (`ONNXTensorRTDetector` + `tensorrt_onnx/YoloONNX`):

1. Захват — `cv::VideoCapture` (FFmpeg/OpenCV backend): кадр в `cv::Mat` на CPU (BGR).
   `VideoExample.cpp:174`, `AsyncDetector.cpp:373`.
2. `frame.getUMat(ACCESS_READ)` — `cv::UMat` используется как ленивая обёртка.
3. `Detect()` сразу вызывает `colorFrame.getMat(cv::ACCESS_READ)` — скачивание D2H.
   `ONNXTensorRTDetector.cpp:192,305`.
4. `ProcessInputAspectRatio` — на CPU: `cv::resize` (letterbox), `cv::split`,
   swap B↔R, `convertTo` float/255. `YoloONNX.cpp:439-564`.
5. `copyInputToDevice()` — H2D. `YoloONNX.cpp:384`.
6. `executeV2()` → `copyOutputToHost()` — D2H. `YoloONNX.cpp:391`.
7. Постобработка `GetResult` — на CPU (в каждом `*_bb.hpp`/`*_instance.hpp`).

Итого: три сквозных перехода через PCIe (кадр D2H, тензор H2D, выходы D2H) +
CPU-препроцессинг. `cv::cuda::GpuMat`/`cudacodec` в проекте не используются.

## 2. Что делает эталон `D-FINE-seg/cpp`

- **Декод** — FFmpeg-демукс + cuvid-парсер + NVDEC, на выходе NV12-поверхности
  в видеопамяти (`src/video/nvdec.cpp`).
- **Препроцесс** — одно fused-ядро `nv12_to_input`: NV12 → RGB → bilinear-resize →
  /255 → NCHW float, пишет прямо в device-буфер входа TRT
  (`src/video/kernels_video.cu:33-55`).
- **Результаты остаются на GPU** (boxes/masks/sem/kpts — device-указатели);
  наружу копируется только `Dets` (~8 КБ) опционально (`include/dfine.h:9-13`).
- **Постобработка (NMS и т.д.) — на GPU**, плюс CUDA-graph для engine
  (`src/core/engine.cpp:88-102`, `src/core/kernels_core.cu`).
- **Разделение «ядро» / «видео»**: engine + ядра (нужны только CUDA + TensorRT)
  собираются отдельно от NVDEC/NVENC-модуля (`CMakeLists.txt:28-56`).

## 3. Ответы на три вопроса

### 3.1 Есть ли смысл?

**Да, но с дисциплиной по объёму и с обязательным замером до кода.** Оценки
порядка величины (не замеренные, для калибровки):

| Устраняемое | Текущее | Оценка |
|---|---|---|
| D2H кадра 1080p BGR (~6 МБ) | `getMat(ACCESS_READ)` | ~0.6–1 мс |
| H2D входного тензора (~5 МБ float) | `copyInputToDevice` | ~0.5 мс |
| CPU resize+split+swap+normalize | `ProcessInputAspectRatio` | ~2–6 мс |
| D2H выходов + CPU-NMS | `copyOutputToHost` + `GetResult` | малый (выходы ~МБ) |
| CPU-декод (если OpenCV без NVDEC) | `cv::VideoCapture` | ~5–25 мс при 1080p |

Суммарно избегаемое — **3–8 мс/кадр** (препроцесс + пересылки), плюс 5–25 мс,
если декод программный. Существенно на 1080p/4K, многопотоке, или когда CPU
делится с трекером. Если модель занимает 15–30 мс, препроцесс остаётся не
«революцией», а снятием накладного расхода.

**Не нужно сразу переносить на GPU постобработку.** Выходы YOLO-семейств малы,
их D2H дёшев, а `GetResult` реализован по-разному для ~15 типов моделей —
перенос всего этого на CUDA — отдельный проект. Максимум пользы за минимум
риска даёт **GPU-препроцесс + отказ от D2H кадра**; постобработку оставить на CPU.

### 3.2 CUDA напрямую или OpenCV `gpuMat`?

- **Препроцесс — напрямую на CUDA (fused-ядро).** OpenCV `cv::cuda` не даёт за
  один проход выход «NCHW float/255 с BGR→RGB и letterbox-паддингом» — либо
  цепочка из 4–5 запусков с промежуточными `GpuMat`, либо всё равно свой
  финальный kernel. Проект уже собирает `yolo_rt_lib` через `cuda_add_library`,
  поэтому один `.cu` — минимальное трение; зависимость от OpenCV-with-CUDA не нужна.
- **Декод — за абстракцией.** `cv::cudacodec::VideoReader` проще, но требует
  OpenCV с модулем cudacodec (в предсобранных сборках обычно нет). Прямой NVDEC
  (Video Codec SDK + FFmpeg) — больше кода, но полный контроль и независимость
  от сборки OpenCV.

### 3.3 Как переделать API, чтобы работало и с CUDA, и без

Паттерн гейтинга уже есть (`BUILD_ONNX_TENSORRT` + `-DBUILD_ONNX_TENSORRT` +
условное включение `ONNXTensorRTDetector.cpp`). Расширяем его:

1. Один макрос `MT_HAS_CUDA` (только в CUDA-сборке). `#include <cuda_runtime.h>`
   и CUDA-типы не попадают в публичные заголовки: в интерфейсах — `void*`/`intptr_t`
   для device-указателей и stream.
2. Тип `GpuFrame` (не зависящий от OpenCV): `{void* data; int w,h; int pitch;
   PixelFormat fmt; intptr_t stream; bool on_device;}`, `fmt ∈ {BGR8, RGB8, NV12, GRAY8}`.
3. `BaseDetector` получает опциональный виртуальный метод
   `Detect(const GpuFrame&, regions_t&)` с дефолтной реализацией «скачать на CPU и
   вызвать обычный `Detect(cv::UMat)`». Переопределяют только детекторы с GPU-входом.
   Полная обратная совместимость: `OCVDNNDetector`, `BackgroundSubtract`,
   `MotionDetector` не трогаем.
4. CMake-опции:
   - `MT_CUDA` (ON подразумевается `BUILD_ONNX_TENSORRT`; включает язык CUDA + `.cu`
     с препроцесс-ядром).
   - `MT_CUDA_VIDEO` (OFF по умолчанию; NVDEC/NVENC-модуль, требует Video Codec SDK
     + FFmpeg).
   - Критерий приёмки: ядро (`mtracking`/`mdetection`) и примеры собираются при
     обоих выключенных.
5. Разделение «ядро» / «видео»: препроцесс/постпроцесс-ядра — внутри детектора;
   NVDEC/NVENC — отдельный опциональный модуль захвата в слое
   `example`/`async_detector`, а не в `BaseDetector`.

## 4. Что в исходной формулировке нужно поправить

1. **«Кадр NV12 не копируется» — не совсем.** NVDEC требует `cuvidUnmapVideoFrame`
   внутри display-callback, поэтому поверхность нельзя отдать детектору напрямую.
   Всегда есть одна D2D-копия в ring-буфер + host-синхронизация на unmap
   (`nvdec.cpp:182-195`, там `cudaStreamSynchronize`). Избегается именно D2H, а не
   всякое копирование.
2. **Letterbox vs squish.** Текущий `ProcessInputAspectRatio` делает letterbox
   (pad `Scalar::all(128)`), а ядро D-FINE-seg — «squish» без letterbox
   (`keep_ratio=False`). GPU-ядро обязано воспроизводить текущую математику
   (letterbox + pad 128 + BGR→RGB + /255 + NCHW), иначе детекции поплывут.
3. **Цветовое пространство.** При входе NV12 нужен выбор BT.601/BT.709 из
   матричных коэффициентов битстрима (`nvdec.cpp:119-120`). Для детекции обычно
   некритично, но для паритета с BGR-путём стоит зафиксировать.
4. **Трекеру кадр на CPU всё равно нужен.** `CTracker::Update` при
   `m_filterGoal == FilterRect` использует CSRT/KCF, читающие изображение на CPU
   (`track.cpp:931,975,1036`), плюс отрисовка/`imshow`/`VideoWriter`. «Полностью
   без копирования в системную память» достижимо только если вывод тоже на GPU
   (NVENC). Реалистичная цель — не скачивать кадр для детекции.
5. **Платформенность D-FINE-seg.** Их видео-модуль жёстко Linux
   (`/usr/include/x86_64-linux-gnu`, `libnvinfer.so.10`, `pkg-config ffmpeg`).
   Проект собирается и на Windows. Портировать NVDEC/NVENC на Windows — отдельный
   объём; начинать с препроцесса (чистый CUDA, кросс-платформенно).
6. **CUDA-graph** — отдельная оптимизация (снижение launch-overhead), не
   обязательная для первого шага, конфликтует с batch>1/динамическими shape.

## 5. План реализации (по фазам)

### Фаза 0 — замер (1–2 дня)

Обязательно до кода. Nsight Systems/`nvtx` или грубые таймеры вокруг decode /
`getMat` / `ProcessInputAspectRatio` / `copyInputToDevice` / `executeV2` /
`copyOutputToHost` / `GetResult` на 1080p и 4K. Подтвердить, что
decode+preprocess+пересылки ≥ порога. Зафиксировать эталонные детекции для
паритета после переноса.

### Фаза 1 — GPU-препроцесс в детекторе (ядро работы, низкий риск)

- В `tensorrt_onnx/` добавить `preprocess.cu` с fused-ядром
  `bgr8_to_input(const uint8_t* src, pitch, w, h, float* dst, in_h, in_w, stream)` —
  letterbox + pad 128 + BGR→RGB + /255 + NCHW, бит-точный к текущему
  `ProcessInputAspectRatio`.
- В `YoloONNX` добавить ветку: при входе device-указателем писать прямо в
  `getDeviceBuffer(input)` и пропустить `copyInputToDevice()`; `Detect` расширить
  перегрузкой, принимающей `GpuFrame`/device-буфер.
- В `ONNXTensorRTDetector` добавить `Detect(const GpuFrame&)`, обходящий
  `getMat(ACCESS_READ)`.
- Критерий: паритет детекций с CPU-путём (тест-скрипт) + замер ускорения.

### Фаза 2 — API/гейтинг

- `GpuFrame`, макрос `MT_HAS_CUDA`, виртуальный `Detect(const GpuFrame&)` в
  `BaseDetector` с CPU-fallback, `MT_CUDA`/`MT_CUDA_VIDEO` в CMake, чистая сборка
  без CUDA в CI.
- Вынести жёстко зашитый `-gencode` список в `CMAKE_CUDA_ARCHITECTURES`
  (`tensorrt_onnx/CMakeLists.txt:30-37`).

### Фаза 3 — NVDEC-декод (опциональный модуль, самый объёмный)

- Отдельный target/модуль `video_capture` по образцу `nvdec.cpp`: FFmpeg-демукс +
  cuvid + ring NV12 в видеопамяти.
- Интеграция в `async_detector`/`VideoExample`: поток захвата владеет декодером +
  ring, поток детекции — TRT-контекстом + stream; релиз слота по event.
- Сначала Linux (эталон готов), потом Windows (загрузка `nvcuvid` DLL).
- NVENC/overlay — только если нужен GPU-вывод видео; иначе кадр один раз D2H для
  трекера/`imshow`.

### Фаза 4 (опционально, отдельно)

GPU-постобработка и CUDA-graph — только если Фазы 0–3 покажут, что
`copyOutputToHost` + CPU-NMS стали узким местом.

**Рекомендуемый порядок вложений:** Фаза 0 → Фаза 1 (+замер) → Фаза 2 → по
результатам решать про Фазу 3. Фазы 1–2 дают большую часть выигрыша при
минимальном риске; NVDEC (Фаза 3) — самый дорогой и платформенно-хрупкий кусок.
