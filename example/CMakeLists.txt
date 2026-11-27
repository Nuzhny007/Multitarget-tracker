cmake_minimum_required (VERSION 3.16)

project(MultitargetTracker)

set(SOURCES main.cpp
            VideoExample.cpp)

set(HEADERS MouseExample.h
            VideoExample.h
            examples.h
            MotionDetectorExample.h
            FileLogger.h)

if (USE_VSR)
    set(SOURCES ${SOURCES} YoloCls.cpp TrackPostProcessor.cpp)
    set(HEADERS ${HEADERS} YoloCls.h TrackPostProcessor.h)
endif(USE_VSR)

if (BUILD_CARS_COUNTING)
    set(SOURCES ${SOURCES} CarsCounting.cpp)
    set(HEADERS ${HEADERS} CarsCounting.h)
endif(BUILD_CARS_COUNTING)

# ----------------------------------------------------------------------------
# добавляем include директории
# ----------------------------------------------------------------------------
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/../src
                    ${PROJECT_SOURCE_DIR}/../src/mtracking
                    ${PROJECT_SOURCE_DIR}/../src/Detector
                    ${PROJECT_SOURCE_DIR}/../src/Detector/vibe_src
                    ${PROJECT_SOURCE_DIR}/../src/Tracker
                    ${PROJECT_SOURCE_DIR}/../src/Tracker/HungarianAlg
                    ${PROJECT_SOURCE_DIR}/../thirdparty
                    ${PROJECT_SOURCE_DIR}/../thirdparty/spdlog/include)

set(LIBS ${OpenCV_LIBS}
         mtracking
         mdetection
         inih
         Threads::Threads)

if (OpenMP_CXX_FOUND)
    set(LIBS ${LIBS} OpenMP::OpenMP_CXX)
endif()

if (BUILD_ONNX_TENSORRT)
    add_definitions(-DBUILD_ONNX_TENSORRT)
endif(BUILD_ONNX_TENSORRT)

if (BUILD_CARS_COUNTING)
    add_definitions(-DBUILD_CARS_COUNTING)
endif(BUILD_CARS_COUNTING)

if (USE_CLIP)
    set(LIBS ${LIBS} ruclip::ruclip)
endif(USE_CLIP)

if (USE_GEOGRAPHICLIB)
    set(LIBS ${LIBS} GeographicLib::GeographicLib)
endif(USE_GEOGRAPHICLIB)

# ---------------------------------------------------------------------------
# Optional classifier/VSR post-processing (demo pipeline).
# Links the vsr_inference static library and the TensorRT/CUDA headers needed
# by the yolo11s-cls TensorRT runner. VSR_INFERENCE_DIR points at the
# cpp/vsr_inference install prefix (include/ + lib/).
# ---------------------------------------------------------------------------
if (USE_VSR)
    add_definitions(-DUSE_VSR)
    set(VSR_INFERENCE_DIR "C:/work/diss/cpp/install" CACHE PATH "vsr_inference install prefix")
    set(TRT_INCLUDE_DIR "C:/work/libraries/cuda/TensorRT-10.16.1.11/include" CACHE PATH "TensorRT include dir")
    set(CUDA_INCLUDE_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/include" CACHE PATH "CUDA include dir")

    include("${VSR_INFERENCE_DIR}/lib/cmake/vsr_inference/vsr_inference_targets.cmake")

    INCLUDE_DIRECTORIES("${VSR_INFERENCE_DIR}/include"
                        "${TRT_INCLUDE_DIR}"
                        "${CUDA_INCLUDE_DIR}")
    set(LIBS ${LIBS} vsr::vsr_inference)
endif(USE_VSR)

ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCES} ${HEADERS})

TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE ${LIBS})

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "apps")