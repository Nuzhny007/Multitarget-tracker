# M2. План архитектурного рефакторинга

Источник: `docs/roadmap.md` §5 (M2). Оценка 1–2 месяца; фазы упорядочены по
риску и зависимостям. Каждая фаза заканчивается зелёной сборкой (и, где можно,
smoke-прогоном), чтобы рефакторинг можно было останавливать после любой фазы.

> **Статус (2026-11-28):** фазы 1–4 выполнены; фаза 5 — опционально;
> фаза 6 — шаг 1 (переходные `mtracking::` алиасы + semver/CHANGELOG);
> перенос определений в `namespace mtracking` и explicit exports — в 2.0.

## Фаза 1. Разделение конфигов и версия схемы INI (M2.1)

**Задача:** отделить настройки детектора от настроек трекера; ввести явную
версию схемы INI.

**Что делается:**
- Новый `DetectorSettings` (`src/mtracking/DetectorSettings.h`): модель, backend,
  пороги, crop/batch/GPU, DNN target/backend, video memory.
- `TrackerSettings` наследует `DetectorSettings` — доступ к полям
  (`settings.m_nnWeights` и т.п.) сохраняется, поэтому существующий код и
  примеры не меняются.
- `ParseTrackerSettings` наполняет базу через `ParseDetectorSettings`; секции
  `[tracking]`/`[detection]` остаются прежними.
- Версия схемы: `[meta] schema_version` (по умолчанию 1), константа
  `kTrackerSettingsSchemaVersion`, предупреждение о неизвестной версии.

**Проверка:** сборка `mtracking`/`mdetection`/`MultitargetTracker`; smoke
`-e=1`; старые INI читаются без изменений.

**Риск:** низкий (наследование сохраняет исходный API).

## Фаза 2. Реестры детекторов и трекеров вместо `switch` (M2.4)

**Задача:** заменить `switch` в `BaseDetector::CreateDetector` и
`BaseTracker::CreateTracker` на реестр фабрик.

**Что делается:**
- `BaseDetector::RegisterDetector(type, factory)` + реестр; встроенные
  детекторы (`Motion_*`, `DNN_OCV`, `ONNX_TensorRT`) регистрируются один раз
  (thread-safe static init).
- `BaseTracker::RegisterTracker(type, factory)`; `UniversalTracker` (`CTracker`)
  и `ByteTrack` регистрируются в `BaseTracker.cpp`.
- Сохраняются прежние коды ошибок/логи (`nullptr`, «not available in this build»,
  проверка `CheckFilterModel`).

**Проверка:** сборка + smoke; недоступные типы по-прежнему дают `nullptr`, а не
падение.

**Риск:** низкий (локальные изменения, поведение сохраняется).

## Фаза 3. Граница публичного API и install (M2.2, часть 1)

**Задача:** стабильный набор публичных заголовков; внутренние типы не
устанавливаются.

**Что делается:**
- Публичные: трекер — `BaseTracker.h`, `TrackerSettings.h`, `DetectorSettings.h`,
  `trajectory.h`, `defines.h`, `object_types.h`; детектор — `BaseDetector.h`.
- Внутренние (не устанавливать): `track.h`, `Kalman.h`, `ShortPathCalculator.h`,
  `EmbeddingsCalculator.hpp`, `HungarianAlg/*`, `LAPJV_algorithm/*`,
  `byte_track/*`, `MotionDetector.h`, `BackgroundSubtract.h`, `vibe_src/*`,
  `OCVDNNDetector.h`, `ONNXTensorRTDetector.h`, TensorRT `*_bb.hpp`.
- `MT_API` для публичных классов + отказ от `WINDOWS_EXPORT_ALL_SYMBOLS` на
  MSVC shared.
- Semver: согласовать `project(MTTracking VERSION …)` с публичными изменениями.

**Проверка:** `cmake --install` + consumer-проект (static и shared).

**Риск:** средний (ABI, упаковка; возможен переходный период).

> **Статус (2026-11-27):** выполнено: публичный набор заголовков, install только
> их, `MT_API` (placeholder: пусто на MSVC из-за export-all, default-visibility
> на GCC), consumer-тест (static) — `CONSUMER_EXIT=0`. Отложено: explicit
> `dllexport`/`dllimport` и отказ от `WINDOWS_EXPORT_ALL_SYMBOLS` (нужна
> проверка shared-сборки на MSVC/Linux).

## Фаза 4. Общие утилиты `example`/`async_detector` (M2.3)

**Задача:** убрать дублирование логгеров, `Frame`-кэшей, таймингов и отрисовки.

**Что делается:**
- Общий модуль (`src/util` или маленькая `mtutil`) для spdlog-sinks, `Frame`
  (Mat/UMat-кэши), таймингов, палитры/`DrawFilledRect`.
- `example` и `async_detector` используют его; дублирующиеся куски удаляются.

**Проверка:** сборка обоих примеров + smoke.

**Риск:** средний (связывание с OpenCV/spdlog, заголовочные зависимости).

> **Статус (2026-11-27):** выполнено: `src/util/{Logger,Palette,Frame}.h`
> (header-only), оба приложения используют их; дублирующиеся блоки удалены.
> Smoke: `MultitargetTracker -e=1` и `AsyncDetector` — `EXIT=0`.

## Фаза 5. TensorRT-слой (M2.5)

**Задача:** абстракция над версиями TensorRT; `common/` вынести из репозитория
или зафиксировать как submodule.

**Что делается:**
- `src/Detector/tensorrt_onnx/common/` (sample-код NVIDIA) — pinned
  submodule/внешняя зависимость; `API.h` изолирует различия версий TRT.
- Проверка `NV_TENSORRT_MAJOR` сосредоточена в одном месте.

> **Статус (2026-11-28):** опционально / не требуется. `common/` — вендорный
> код NVIDIA (38 файлов, Apache-2.0), уже в отдельной библиотеке `yolo_rt_lib`;
> 70 из 74 `NV_TENSORRT_MAJOR` — в нём, в нашем коде 4 ветки `< 8`. Вынос в
> `thirdparty/`/submodule — лицензионная гигиена, не архитектура; версионная
> абстракция не нужна, пока поддерживается одна версия TRT.

**Проверка:** сборка с `BUILD_ONNX_TENSORRT=ON` (нужен TensorRT) + прогон.

**Риск:** высокий (внешние зависимости, self-hosted runner).

## Фаза 6. `namespace mtracking` + семантическое версионирование (M2.2, часть 2)

**Задача:** стабильный неймспейс и явный мажор при breaking-изменениях.

**Что делается:**
- Публичные типы (`BaseTracker`, `TrackingObject`, `CRegion`, `TrackerSettings`,
  `objtype_t`, `track_id_t`, …) в `namespace mtracking` с переходными
  `using`-алиасами.
- `MTTracking VERSION 2.0.0`, CHANGELOG, обновление README/примеров/python.

**Проверка:** сборка, python-биндинги, consumer; smoke.

**Риск:** высокий (breaking API) — делать в отдельной ветке/мажорном релизе.

> **Статус (2026-11-28):** шаг 1 выполнен — `mtracking/Mtracking.h` с
> переходными `using`-алиасами (`mtracking::BaseTracker`, `TrackerSettings`,
> `TrackingObject`, `CRegion`, …), `mtracking::tracking` для enum'ов,
> `MTTracking 1.2.0` + `CHANGELOG.md`; consumer с `mtracking::` собран и
> запущен (`EXIT=0`). Отложено (2.0): перенос определений в namespace,
> explicit `MT_API` exports/отказ от `WINDOWS_EXPORT_ALL_SYMBOLS`.

## Порядок и критерии остановки

1. Фаза 1 → Фаза 2 (низкий риск, без изменения поведения).
2. Фаза 3 (публичный API/install) — после стабилизации реестров.
3. Фаза 4 (общие утилиты).
4. Фаза 5/6 — по готовности внешних зависимостей и с отдельной веткой
   для breaking-изменений.

Остановка допустима после любой фазы: каждая оставляет сборку зелёной и не
ломает существующий API.
